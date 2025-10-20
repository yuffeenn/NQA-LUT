import json
from tabulate import tabulate


def parse_qpwl(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    func_name = data['func_name']
    mse = data['mse']
    mae = data['mae']
    return [func_name, f"{mae:.3e}", f"{mse:.3e}"]


def batch_parse(json_files):
    table_data = []
    for json_file in json_files:
        table_data.append(parse_qpwl(json_file))

    headers = ["Function", "MAE", "MSE"]
    print(tabulate(table_data, headers=headers, floatfmt=("", ".3e", ".3e"), tablefmt="grid"))


if __name__ == "__main__":
    json_files = [
        'table1/log/sigmoid/qpwl_e10b10.json',
        'table1/log/gelu/qpwl_e7b11.json',
        'table1/log/elu/qpwl_e10b9.json',
        'table1/log/silu/qpwl_e10b11.json',
        'table1/log/tanh/qpwl_e10b10.json'
    ]
    batch_parse(json_files)

    json_files = [
        'table2/log_e8/gelu/qpwl_e8b8.json',
        'table2/log_e8/hswish/qpwl_e8b8.json',
        'table2/log_e8/exp/qpwl_e8b8.json',
        'table2/log_e8/div/qpwl_e8b8.json',
        'table2/log_e8/rsqrt/qpwl_e8b8.json'
    ]
    batch_parse(json_files)

    json_files = [
        'table2/log_e16/gelu/qpwl_e16b8.json',
        'table2/log_e16/hswish/qpwl_e16b8.json',
        'table2/log_e16/exp/qpwl_e16b8.json',
        'table2/log_e16/div/qpwl_e16b8.json',
        'table2/log_e16/rsqrt/qpwl_e16b8.json'
    ]
    batch_parse(json_files)
