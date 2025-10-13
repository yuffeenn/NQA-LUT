import torch
import torch.nn as nn
import torch.optim as optim
import json
from relunn import ReluNN
from utils import round_ste, ceil_ste, setup_logging


class QPWL(ReluNN):
    def __init__(self, num_entries=8, x_range=(-5.0, 5.0), func_name='gelu', ckpt_path=None, bits=8):
        super().__init__(num_entries, x_range, func_name)
        self.bits = bits
        self.bit_max = 2. ** (bits - 1)
        self.intervals = []
        # load state
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.load_state_dict(ckpt)
        self.parm_k = nn.ParameterList([nn.Parameter(k, requires_grad=True) for k in self.pwl_k])
        self.parm_b = nn.ParameterList([nn.Parameter(b, requires_grad=True) for b in self.pwl_b])
        self.layer.requires_grad_(False)
        self.points = None
        self.piece = None
        self.point_min_interval = None

    def gen_intervals(self):
        edges = torch.cat([
            torch.tensor([self.x_range[0]], device=self.pwl_p.device),
            self.pwl_p,
            torch.tensor([self.x_range[1]], device=self.pwl_p.device)
        ])
        self.intervals = [(edges[i].item(), edges[i + 1].item()) for i in range(len(edges) - 1)]

    def get_qpwl_params(self):
        point_ints, point_shift = self.shift_scaling_quantize(self.pwl_p, False)
        min_interval = (point_ints[1:] - point_ints[:-1]).min()
        self.point_min_interval = int(min_interval)
        self.points = {'int_vals': point_ints.int().tolist(), 'shift': int(point_shift)}
        pwl_kb = torch.stack((self.pwl_k, self.pwl_b), dim=1)
        self.piece = []
        for idx in range(self.num_entries):
            piece_int, piece_shift = self.shift_scaling_quantize(pwl_kb[idx], False)
            self.piece.append({'k': int(piece_int[0]), 'b': int(piece_int[1]), 'shift': int(piece_shift)})

    def shift_scaling_quantize(self, values, fake_quant=True):
        max_abs = values.abs().max()
        if max_abs < 1e-12:
            return values if fake_quant else (torch.zeros_like(values), 0)

        shift = ceil_ste(torch.log2(max_abs / self.bit_max))
        scale = 2.0 ** shift
        quantized = round_ste(values / scale)
        quantized = torch.clamp(quantized, -self.bit_max, self.bit_max -1)

        return quantized * scale if fake_quant else (quantized, -shift)

    def train_one_interval(self, logger, idx, total_epochs=5000, lr=1e-3):
        logger.info(f"--- Training Interval[{idx}]: {self.intervals[idx]} ---")
        device = next(self.parameters()).device
        criterion = nn.L1Loss()

        optimizer = optim.AdamW([self.parm_k[idx], self.parm_b[idx]], lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

        left, right = self.intervals[idx]
        x = torch.linspace(left, right, 1000, device=device)
        y_true = self.target_func(x)

        best_loss = float('inf')
        best_k, best_b = None, None
        for epoch in range(total_epochs):
            # pwl_k = self.shift_scaling_quantize(self.parm_k[idx])
            # pwl_b = self.shift_scaling_quantize(self.parm_b[idx])
            pwl_k, pwl_b = self.shift_scaling_quantize(torch.stack((self.parm_k[idx], self.parm_b[idx]))).unbind(0)
            y_pred = pwl_k * x + pwl_b
            loss = criterion(y_pred, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_k = pwl_k.clone().detach()
                best_b = pwl_b.clone().detach()

            if epoch % 500 == 0 or epoch == total_epochs - 1:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"  Interval {idx}, Epoch {epoch:4d}, Loss: {loss.item():.6f}, LR: {current_lr:.2e}")
        self.parm_k[idx].data = best_k
        self.parm_b[idx].data = best_b

    def qat(self, total_epochs=5000, lr=1e-3):
        save_name = f'log/{self.func_name}/qpwl_{self.num_entries}entry'
        logger = setup_logging(f'{save_name}.log')
        self.pwl_p = self.shift_scaling_quantize(self.pwl_p)
        self.gen_intervals()

        for idx in range(self.num_entries):
            self.train_one_interval(logger, idx, total_epochs=total_epochs, lr=lr)
        self.pwl_k = torch.stack(list(self.parm_k)).flatten()
        self.pwl_b = torch.stack(list(self.parm_b)).flatten()
        torch.save(self.state_dict(), f"{save_name}.pt")

        self.visual(save_path=f'{save_name}.png')
        mae = self.evaluate()
        logger.info(f"Final MAE:  {mae:.6f}")

        self.get_qpwl_params()
        deploy_params = {
            'func_name': self.func_name,
            'num_entries': self.num_entries,
            'x_range': list(self.x_range),
            'bits': self.bits,
            "total_epochs": total_epochs,
            "lr": lr,
            'mae': mae,
            'point_min_interval': self.point_min_interval,
            'point': self.points,
            'piece': self.piece
        }
        json_path = f"{save_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(deploy_params, f, indent=2, ensure_ascii=False)
        logger.info(f"Deployable parameters saved to: {json_path}")


if __name__ == "__main__":
    qpwl = QPWL(8, (-5.0, 5.0), 'gelu', 'log/gelu_8entry.pt', 8)
    qpwl.qat(total_epochs=5000, lr=1e-3)
