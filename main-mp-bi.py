import os
import json
import time
import math
import multiprocessing
from functools import reduce
from argparse import ArgumentParser

from transformers import set_seed
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from utils import add_args, build_dataset


def worker(kwargs):
    # args
    args = kwargs['args']
    data = kwargs['data']
    device = kwargs['device']

    # envs
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    set_seed(args.seed)

    # prepare model and generation config
    llm = LLM(
        seed=args.seed,
        model=args.model_name_or_path, 
        trust_remote_code=True, 
        tensor_parallel_size=1, 
        # gpu_memory_utilization=0.2, 
        max_model_len=args.cutoff_len, 
        enforce_eager=False,
        enable_lora=bool(args.lora_adapter_path),
    )
    sampling_params = SamplingParams(
        temperature=args.temperature, 
        top_p=args.top_p, 
        max_tokens=args.max_new_tokens, 
    )

    # do generate!
    preds = []
    prompts = list(map(lambda x: x['llm_prompt'], data))
    outputs = llm.generate(
        prompts, 
        sampling_params=sampling_params, 
        use_tqdm=True, 
        lora_request=LoRARequest('lora', 1, args.lora_adapter_path) if args.lora_adapter_path else None
    )
    for i, out in enumerate(outputs):
        preds.append({
            "prompt": data[i]['llm_prompt'], 
            "predict": out.outputs[0].text,
            "label": data[i]['label']
        })
    
    return preds


def main(args):
    data = build_dataset(args)

    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    device_cnt = len(devices)
    data_per_device_len = math.ceil(len(data) / device_cnt)
    with multiprocessing.Pool(device_cnt) as pool:
        worker_arguments = [
            {
                "args": args, 
                "data": data[
                    i * data_per_device_len: 
                    min((i+1) * data_per_device_len, len(data))
                ], 
                "device": device
            }
            for i, device in enumerate(devices)
        ]
        all_preds = pool.map(worker, worker_arguments)

    all_preds = reduce(lambda x, y: x+y, all_preds, [])
    print(f'{len(all_preds) = }', flush=True)
    with open(f"{args.output_dir}/generated_predictions.jsonl", "w", encoding="utf-8") as f_pred:
        for entry in all_preds:
            f_pred.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    
    args.model_name_or_path = os.path.realpath(args.model_name_or_path)
    args.lora_adapter_path = os.path.realpath(args.lora_adapter_path)
    args.dataset_dir = os.path.realpath(args.dataset_dir)
    args.output_dir = os.path.realpath(args.output_dir)
    args.multistep_gen_config = os.path.realpath(args.multistep_gen_config) if args.multistep_gen_config else None
    args.tree_instance = os.path.realpath(args.tree_instance) if args.tree_instance else None
    
    start_time = time.time()
    main(args)
    end_time = time.time()
    print(f'Consuming time: {(end_time - start_time)}s', flush=True)
