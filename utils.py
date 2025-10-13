import torch
from torch import nn
import os
import logging


def get_target_function(func_name):
    func_dict = {
        'gelu': nn.functional.gelu,
        'exp': torch.exp,
        'div': lambda x: 1 / x,
        'sigmoid': torch.sigmoid,
        'tanh': torch.tanh,
        'silu': nn.functional.silu,
        'elu': nn.functional.elu,
        'hardsigmoid': nn.functional.hardsigmoid,
        'hardswish': nn.functional.hardswish,
    }

    if func_name not in func_dict:
        raise ValueError(f"Unsupported function: {func_name}. "
                         f"Supported functions are: {list(func_dict.keys())}")
    return func_dict[func_name]


def setup_logging(save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(save_path, encoding='utf-8', mode='w')
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def convert_tensor_to_list(tensor):
    return tensor.detach().cpu().numpy().tolist() if tensor is not None else None


class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = input.round()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Ceil(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.ceil(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


round_ste = Round.apply
ceil_ste = Ceil.apply
