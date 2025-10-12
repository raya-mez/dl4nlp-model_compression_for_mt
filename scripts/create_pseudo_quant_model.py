import argparse
import os

from model import load_model_and_tokenizer, replace_linear_with_quant, save_quant_to_safetensors
from utils import KBitConfig
from constants import max_allowed_torch_dtype


def load_and_save_k_bit_quantized_model(k, model_id_key, precision, kbit_cfg_group_size, pseudo_quant_base_dir):
    cfg = KBitConfig(k=k, group_size=kbit_cfg_group_size, f_dtype=max_allowed_torch_dtype)
    base_model, tokenizer = load_model_and_tokenizer(model_id_key, precision)
    quantized_model = replace_linear_with_quant(base_model, cfg)

    save_dir = os.path.join(pseudo_quant_base_dir, f'{k:02d}_bit_model')
    save_quant_to_safetensors(quantized_model, str(save_dir))
    del base_model, quantized_model, tokenizer


def main(arguments):
    pseudo_quant_base_dir = os.path.join(arguments.result_base_dir, 'pseudo_quant_models')
    os.makedirs(pseudo_quant_base_dir, exist_ok=True)

    if arguments.run_for_k_range:
        for k in range(1, arguments.target_k + 1):
            load_and_save_k_bit_quantized_model(k, arguments.model_id_key, arguments.precision,
                                                arguments.kbit_cfg_group_size, pseudo_quant_base_dir)
    else:
        load_and_save_k_bit_quantized_model(arguments.target_k, arguments.model_id_key, arguments.precision,
                                            arguments.kbit_cfg_group_size, pseudo_quant_base_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id_key', default='tower_instruct', type=str,
                        choices=['tower_instruct', 'llama_chat'])
    parser.add_argument('--precision', default='fp16',
                        type=str, help='Precision for which scores are being calculated if not Pseudo quantization',
                        choices=['fp16', 'bf16', 'int8', 'int4'])
    parser.add_argument('--kbit_cfg_group_size', default=64,
                        type=int, help='Group Size for KBitConfig')
    parser.add_argument('--run_for_k_range', action='store_true',
                        help='If set, means run for range 1 to target_k')
    parser.add_argument('--target_k', default=16,
                        type=int, help='target_k value. When running on entire range, it becomes the upper bound.')
    parser.add_argument('--result_base_dir', default='../results/', type=str)
    args = parser.parse_args()
    main(args)
