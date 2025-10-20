import json
import torch
from relunn import ReluNN
from utils import round_ste, ceil_ste
from tabulate import tabulate


class PPWL(ReluNN):
    def __init__(self, num_entries=8, x_range=(-5.0, 5.0), func_name='gelu', ckpt_path=None, bits=8):
        super().__init__(num_entries, x_range, func_name)
        self.bits = bits
        self.bit_max = 2.0 ** (bits - 1)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.load_state_dict(ckpt)
        self.ptq()

    def _gen_intervals(self):
        edges = torch.cat([
            torch.tensor([self.x_range[0]], device=self.pwl_p.device),
            self.pwl_p,
            torch.tensor([self.x_range[1]], device=self.pwl_p.device)
        ])
        self.intervals = [(edges[i].item(), edges[i + 1].item()) for i in range(len(edges) - 1)]

    def shift_scaling_quantize(self, x, fake_quant=True):
        max_abs = x.abs().max().clamp(min=1e-6)
        shift = ceil_ste(torch.log2(max_abs / self.bit_max))
        scale = 2.0 ** shift
        q = round_ste(x / scale)
        q = torch.clamp(q, -self.bit_max, self.bit_max - 1)
        return q * scale if fake_quant else (q.int(), -shift.item())

    def ptq(self):
        self.pwl_p = self.shift_scaling_quantize(self.pwl_p)
        self._gen_intervals()
        pwl_k, pwl_b = [], []
        for i in range(self.num_entries):
            kq, bq = self.shift_scaling_quantize(torch.stack([self.pwl_k[i], self.pwl_b[i]]))
            pwl_k.append(kq)
            pwl_b.append(bq)
        self.pwl_k = torch.tensor(pwl_k)
        self.pwl_b = torch.tensor(pwl_b)


def eval_pwl_ptq(json_path, ckpt_path, bits=8):
    """Evaluate PTQ model and return results as list"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    config = data['config']
    func_name = config['function']
    ppwl = PPWL(
        num_entries=config['num_entries'],
        x_range=config['x_range'],
        func_name=func_name,
        ckpt_path=ckpt_path,
        bits=bits
    )

    mae, mse = ppwl.evaluate()

    return [func_name, f"{mae:.3e}", f"{mse:.3e}"]


def batch_eval(prefixes, bits):
    table_data = []
    for prefix in prefixes:
        result = eval_pwl_ptq(f'{prefix}.json', f'{prefix}.pt', bits)
        table_data.append(result)
    headers = ["Function", "MAE", "MSE"]
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=("", ".3e", ".3e"), stralign="center"))


if __name__ == "__main__":
    bits = 8
    prefixes = [
        'table1/log/sigmoid/10entry',
        'table1/log/gelu/7entry',
        'table1/log/elu/10entry',
        'table1/log/silu/10entry',
        'table1/log/tanh/10entry'
    ]
    batch_eval(prefixes, bits)

    prefixes = [
        'table2/log_e8/gelu/8entry',
        'table2/log_e8/hswish/8entry',
        'table2/log_e8/exp/8entry',
        'table2/log_e8/div/8entry',
        'table2/log_e8/rsqrt/8entry',
    ]
    batch_eval(prefixes, bits)

    prefixes = [
        'table2/log_e16/gelu/16entry',
        'table2/log_e16/hswish/16entry',
        'table2/log_e16/exp/16entry',
        'table2/log_e16/div/16entry',
        'table2/log_e16/rsqrt/16entry',
    ]
    batch_eval(prefixes, bits)
