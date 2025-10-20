import torch
import yaml
from relunn import ReluNN
from qpwl import QPWL


def main(cfg_path):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Step 1: Training ReluNN
    if config['train']:
        print("=== Step 1: Training ReluNN ===")
        model = ReluNN(
            num_entries=config['num_entries'],
            x_range=tuple(config['x_range']),
            func_name=config['func_name'],
            criterion_name=config['criterion_name']
        )
        model.train_model(
            total_epochs=config['train_epochs'],
            warmup_epochs=1000,
            base_lr=config['train_lr'],
            weight_decay=config['weight_decay'],
            range_weight=config['range_weight'],
            prox_weight=config['prox_weight']
        )

    # Step 2: QAT with QPWL
    if config['qat']:
        print("\n=== Step 2: QAT Training with QPWL ===")
        ckpt_path = f"log/{config['func_name']}/{config['num_entries']}entry.pt"
        qpwl_model = QPWL(
            num_entries=config['num_entries'],
            x_range=tuple(config['x_range']),
            func_name=config['func_name'],
            criterion_name=config['criterion_name'],
            ckpt_path=ckpt_path,
            bits=config['bits'],
        )

        qpwl_model.qat(
            total_epochs=config['qat_epochs'],
            lr=config['qat_lr']
        )


if __name__ == "__main__":
    main("config/gelu.yaml")