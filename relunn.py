import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.optim.lr_scheduler import SequentialLR
import json
import matplotlib.pyplot as plt
import numpy as np
from utils import get_target_function, get_loss_criterion, convert_tensor_to_list, setup_logging


class ReluNN(nn.Module):
    def __init__(self, num_entries=8, x_range=(-5.0, 5.0), func_name='gelu', criterion_name='mae'):
        super(ReluNN, self).__init__()
        self.num_entries = num_entries
        self.x_range = x_range
        self.func_name = func_name
        self.target_func = get_target_function(func_name)
        self.criterion = get_loss_criterion(criterion_name)
        self.layer = nn.Sequential(
            nn.Linear(1, self.num_entries - 1, bias=True),
            nn.ReLU(),
            nn.Linear(self.num_entries - 1, 1, bias=True)
        )
        self.pwl_p = None
        self.pwl_k = None
        self.pwl_b = None
        self.init_weight()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict, assign)
        self.convert2lut()
        return result

    @torch.no_grad()
    def init_weight(self):
        H = self.num_entries - 1
        x_min, x_max = self.x_range
        knots = np.linspace(x_min, x_max, H + 2)[1:-1]
        x_nodes = np.concatenate([[x_min], knots, [x_max]])
        y_nodes = self.target_func(torch.from_numpy(x_nodes)).numpy()

        slopes = np.diff(y_nodes) / np.diff(x_nodes)
        delta_s = np.diff(slopes)
        gamma = delta_s
        b_out = y_nodes[0]

        self.layer[0].weight.fill_(1.0)
        self.layer[0].bias.copy_(torch.from_numpy(-knots).float())

        self.layer[2].weight.copy_(torch.from_numpy(gamma).float().view(1, -1))
        self.layer[2].bias.fill_(b_out)

    def range_loss(self):
        breakpoints = -self.layer[0].bias / self.layer[0].weight.squeeze()
        width = self.x_range[1] - self.x_range[0]
        lower_bound = self.x_range[0] + width * 0.005
        upper_bound = self.x_range[1] - width * 0.005
        lower_bound_violation = torch.relu(lower_bound - breakpoints)
        upper_bound_violation = torch.relu(breakpoints - upper_bound)
        constraint_loss = torch.sum(lower_bound_violation + upper_bound_violation)

        return constraint_loss

    def proximity_loss(self, min_internal_factor=32):
        breakpoints = -self.layer[0].bias / self.layer[0].weight.squeeze()
        n = breakpoints.size(0)
        min_distance = (self.x_range[1] - self.x_range[0]) / (min_internal_factor - 0.5)

        triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=breakpoints.device), diagonal=1)
        diff_matrix = breakpoints.unsqueeze(0) - breakpoints.unsqueeze(1)
        abs_diff = torch.abs(diff_matrix)
        upper_tri_diff = abs_diff[triu_mask]
        loss = torch.sum(torch.relu(min_distance - upper_tri_diff))
        return loss

    def convert2lut(self):
        ni = self.layer[0].weight.squeeze()
        bi = self.layer[0].bias
        mi = self.layer[2].weight.squeeze()
        breakpoints = -bi / ni
        sorted_indices = torch.argsort(breakpoints)
        breakpoints = breakpoints[sorted_indices]
        ni = ni[sorted_indices]
        bi = bi[sorted_indices]
        mi = mi[sorted_indices]

        pwl_si = []
        pwl_ti = []

        # interval(1)
        si = torch.sum(mi * ni * (ni < 0))
        ti = torch.sum(mi * bi * (ni < 0))
        pwl_si.append(si.item())
        pwl_ti.append(ti.item())
        # interval(2,N-1)
        for i in range(len(breakpoints) - 1):
            si = torch.sum(mi[:i + 1] * ni[:i + 1] * (ni[:i + 1] >= 0)) + torch.sum(
                mi[i + 1:] * ni[i + 1:] * (ni[i + 1:] < 0))
            ti = torch.sum(mi[:i + 1] * bi[:i + 1] * (ni[:i + 1] >= 0)) + torch.sum(
                mi[i + 1:] * bi[i + 1:] * (ni[i + 1:] < 0))
            pwl_si.append(si.item())
            pwl_ti.append(ti.item())
        # interval(N)
        si = torch.sum(mi * ni * (ni >= 0))
        ti = torch.sum(mi * bi * (ni >= 0))
        pwl_si.append(si.item())
        pwl_ti.append(ti.item())

        self.pwl_p = breakpoints
        self.pwl_k = torch.tensor(pwl_si)
        self.pwl_b = torch.tensor(pwl_ti) + self.layer[2].bias

    def pwl_forward(self, x):
        interval_indices = torch.searchsorted(self.pwl_p, x, right=False)
        zi = self.pwl_k[interval_indices] * x + self.pwl_b[interval_indices]
        return zi

    def forward(self, x):
        return self.layer(x)

    @torch.no_grad()
    def evaluate(self):
        device = next(self.parameters()).device
        x_eval = torch.linspace(self.x_range[0], self.x_range[1], 1000, device=device)
        y_true_eval = self.target_func(x_eval)
        y_pred_eval = self.pwl_forward(x_eval)
        mae = nn.functional.l1_loss(y_pred_eval, y_true_eval).item()
        mse = nn.functional.mse_loss(y_pred_eval, y_true_eval).item()
        return mae, mse

    def train_model(self, total_epochs=20000, warmup_epochs=1000,
                    base_lr=1e-3, weight_decay=1e-4, range_weight=0.1, prox_weight=0.0):
        save_name = f'log/{self.func_name}/{self.num_entries}entry'
        logger = setup_logging(f'{save_name}.log')
        device = next(self.parameters()).device
        x = torch.linspace(self.x_range[0], self.x_range[1], 1000).unsqueeze(1).to(device)
        y_true = self.target_func(x)

        optimizer = optim.AdamW(self.parameters(), lr=base_lr, weight_decay=weight_decay)
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-5)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

        train_config = {
            "function": self.func_name,
            "x_range": self.x_range,
            "num_entries": self.num_entries,
            "total_epochs": total_epochs,
            "warmup_epochs": warmup_epochs,
            "base_lr": base_lr,
            "weight_decay": weight_decay,
            "range_weight": range_weight,
            "prox_weight": prox_weight,
            "device": str(device)
        }
        logger.info("Training Configuration:")
        logger.info(json.dumps(train_config, indent=2))

        best_loss = float('inf')
        best_state_dict = None

        self.train()
        for epoch in range(total_epochs):
            y_pred = self(x)
            main_loss = self.criterion(y_pred, y_true)
            range_loss = range_weight * self.range_loss()
            prox_loss = prox_weight * self.proximity_loss()
            total_loss = main_loss + range_loss + prox_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            scheduler.step()
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_state_dict = self.state_dict().copy()

            if epoch % 500 == 0 or epoch == total_epochs - 1:
                current_lr = optimizer.param_groups[0]['lr']
                log_msg = (f'Epoch [{epoch}/{total_epochs}], '
                           f'Total Loss: {total_loss.item():.3e}, '
                           f'Main Loss: {main_loss.item():.3e}, '
                           f'Range Loss: {range_loss.item():.3e}, '
                           f'Prox Loss: {prox_loss.item():.3e}, '
                           f'LR: {current_lr:.3e}')
                logger.info(log_msg)

        self.load_state_dict(best_state_dict)
        self.convert2lut()

        mae, mse = self.evaluate()

        torch.save(best_state_dict, f"{save_name}.pt")
        train_results = {
            "config": train_config,
            "mae": mae,
            "mse": mse,
            "lookup_table": {
                "breakpoints": convert_tensor_to_list(self.pwl_p),
                "pwl_si": convert_tensor_to_list(self.pwl_k),
                "pwl_ti": convert_tensor_to_list(self.pwl_b)
            }
        }
        with open(f"{save_name}.json", 'w') as f:
            json.dump(train_results, f, indent=2)

        logger.info(f"MAE: {mae:.10f}, MSE: {mse:.10f},")
        logger.info(f"Model saved to {save_name}.pt")
        logger.info(f"Training results saved to {save_name}.json")
        self.visual()

        return best_loss

    @torch.no_grad()
    def visual(self, save_path=None):
        if save_path is None:
            save_path = f'log/{self.func_name}/{self.num_entries}entry.png'

        device = next(self.parameters()).device
        x = torch.linspace(self.x_range[0], self.x_range[1], 1000, device=device)

        y_true = self.target_func(x.unsqueeze(1)).squeeze()
        y_approx = self.pwl_forward(x.unsqueeze(1)).squeeze()

        x_np, y_true_np, y_approx_np = [t.cpu().numpy() for t in [x, y_true, y_approx]]
        error = np.abs(y_true_np - y_approx_np)

        if self.pwl_p is None:
            self.convert2lut()
        breakpoints = self.pwl_p.cpu().numpy()
        k_vals = self.pwl_k.cpu().numpy()
        b_vals = self.pwl_b.cpu().numpy()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Function Approximation
        axes[0, 0].plot(x_np, y_true_np, 'k-', label=f'True {self.func_name.upper()}', linewidth=2)
        axes[0, 0].plot(x_np, y_approx_np, 'r--', label='PWL Approximation', linewidth=1.5)
        axes[0, 0].set_title('Function Approximation')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Piecewise Linear Segments
        full_breakpoints = np.concatenate([[self.x_range[0]], breakpoints, [self.x_range[1]]])
        for i in range(self.num_entries):
            start, end = full_breakpoints[i], full_breakpoints[i + 1]
            k, b = k_vals[i], b_vals[i]
            x_seg = np.linspace(start, end, 200)
            y_seg = k * x_seg + b
            axes[0, 1].plot(x_seg, y_seg, label=f'Segment {i + 1}: $k_{{{i + 1}}}={k:.3f}, b_{{{i + 1}}}={b:.3f}$')
        axes[0, 1].set_title('Piecewise Linear Segments', fontsize=14)
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)

        # Approximation Error
        mae, max_ae = np.mean(error), np.max(error)
        axes[1, 0].plot(x_np, error, 'C3-', label='Absolute Error', linewidth=1.5)
        axes[1, 0].axhline(mae, color='C2', linestyle='--', label=f'MAE = {mae:.4e}')
        axes[1, 0].axhline(max_ae, color='red', linestyle=':', label=f'Max AE = {max_ae:.4e}')
        axes[1, 0].set_title('Approximation Error')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Breakpoint Locations
        bkp_ys = self.pwl_forward(self.pwl_p.unsqueeze(1)).squeeze().cpu().numpy()
        axes[1, 1].plot(x_np, y_true_np, 'k-', alpha=0.7, label='True Function')
        axes[1, 1].scatter(breakpoints, bkp_ys, color='red', s=40, zorder=5, label='Breakpoints')
        for i, (b, y) in enumerate(zip(breakpoints, bkp_ys)):
            axes[1, 1].annotate(f'{b:.2f}', (b, y), xytext=(0, 8),
                                textcoords='offset points', ha='center', fontsize=7)
        axes[1, 1].set_title('Breakpoint Locations')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()

