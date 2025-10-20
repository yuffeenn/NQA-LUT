import torch
import torch.nn as nn
import torch.optim as optim
import json
from relunn import ReluNN
from utils import round_ste, ceil_ste, setup_logging


class QPWL(ReluNN):
    def __init__(self, num_entries=8, x_range=(-5.0, 5.0), func_name='gelu', criterion_name='mae',
                 ckpt_path=None, bits=8):
        super().__init__(num_entries, x_range, func_name, criterion_name)
        self.bits = bits
        self.bit_max = 2. ** (bits - 1)
        self.intervals = []
        # load state
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.load_state_dict(ckpt)
        self.layer.requires_grad_(False)
        # qat init
        self.parm_k = nn.ParameterList()
        self.parm_b = nn.ParameterList()
        self.qat_init()

        self.points = None
        self.piece = None
        self.point_min_interval = None

    def qat_init(self):
        self.pwl_p = self.shift_scaling_quantize(self.pwl_p)
        self.gen_intervals()
        for i in range(self.num_entries):
            x1, x2 = self.intervals[i]
            x1, x2 = torch.tensor(x1), torch.tensor(x2)
            y2 = self.target_func(x2)
            y1 = self.target_func(x1)
            k = (y2 - y1) / (x2 - x1)
            b = y1 - k * x1
            self.parm_k.append(nn.Parameter(k))
            self.parm_b.append(nn.Parameter(b))

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
        max_abs = values.abs().max().clamp(min=0.0625)
        shift = ceil_ste(torch.log2(max_abs / self.bit_max))
        scale = 2.0 ** shift
        quantized = round_ste(values / scale)
        quantized = torch.clamp(quantized, -self.bit_max, self.bit_max - 1)

        return quantized * scale if fake_quant else (quantized, -shift)

    def train_one_interval(self, logger, idx, total_epochs=5000, lr=1e-4):
        logger.info(f"--- Training Interval[{idx}]: {self.intervals[idx]} ---")
        device = next(self.parameters()).device

        optimizer = optim.AdamW([self.parm_k[idx], self.parm_b[idx]], lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

        left, right = self.intervals[idx]
        x = torch.linspace(left, right, 1000, device=device)
        y_true = self.target_func(x)

        best_loss = float('inf')
        best_k, best_b = None, None
        for epoch in range(total_epochs):
            pwl_k, pwl_b = self.shift_scaling_quantize(torch.stack((self.parm_k[idx], self.parm_b[idx]))).unbind(0)
            y_pred = pwl_k * x + pwl_b
            loss = self.criterion(y_pred, y_true)
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
                logger.info(f"  Interval {idx}, Epoch {epoch:4d}, Loss: {loss.item():.3e}, LR: {current_lr:.2e}")
        self.parm_k[idx].data = best_k
        self.parm_b[idx].data = best_b

    def qat(self, total_epochs=5000, lr=1e-3):
        save_name = f'log/{self.func_name}/qpwl_e{self.num_entries}b{self.bits}'
        logger = setup_logging(f'{save_name}.log')

        for idx in range(self.num_entries):
            self.train_one_interval(logger, idx, total_epochs=total_epochs, lr=lr)
        self.pwl_k = torch.stack(list(self.parm_k)).flatten()
        self.pwl_b = torch.stack(list(self.parm_b)).flatten()
        torch.save(self.state_dict(), f"{save_name}.pt")

        self.visual(save_path=f'{save_name}.png')
        mae, mse = self.evaluate()
        logger.info(f"Final MAE: {mae:.3e}, MSE: {mse:.3e}")

        self.get_qpwl_params()
        deploy_params = {
            'func_name': self.func_name,
            'num_entries': self.num_entries,
            'x_range': list(self.x_range),
            'bits': self.bits,
            "total_epochs": total_epochs,
            "lr": lr,
            'mae': mae,
            'mse': mse,
            'point_min_interval': self.point_min_interval,
            'point': self.points,
            'piece': self.piece
        }
        json_path = f"{save_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(deploy_params, f, indent=2, ensure_ascii=False)
        logger.info(f"Deployable parameters saved to: {json_path}")

