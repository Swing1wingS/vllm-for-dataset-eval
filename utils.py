import os, json
from tqdm import tqdm

NAME2TEMPLATE = {
    'empty': {
        'prefix': '', 
        'suffix': '', 
    }, 
    'llama3': {
        'prefix': '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n', 
        'suffix': '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n', 
    }, 
    'qwen': {
        'prefix': '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n', 
        'suffix': '<|im_end|>\n<|im_start|>assistant\n', 
    }, 
}

def add_args(parser):
    # vllm config
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--preprocessing_num_workers', type=int, required=True)
    parser.add_argument('--lora_adapter_path', type=str, default=None)
    parser.add_argument('--template', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--eval_dataset', type=str, required=True)
    parser.add_argument('--cutoff_len', type=int, required=True)
    parser.add_argument('--max_new_tokens', type=int, required=True)
    parser.add_argument('--top_p', type=float, required=True)
    parser.add_argument('--temperature', type=float, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    # custom config
    parser.add_argument('--multistep_gen_config', type=str, default=None)
    parser.add_argument('--tree_instance', type=str, default=None)
    return parser


def prepare_prompt(tokenizer, prompt):
    return prompt


def build_dataset(args):
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    template = NAME2TEMPLATE[args.template]

    dataset_info = json.load(open(os.path.join(args.dataset_dir, 'dataset_info.json')))
    eval_dataset_file = dataset_info[args.eval_dataset]['file_name']
    with open(eval_dataset_file) as f:
        raw_data = json.load(f)
    data = [
        {
            'llm_prompt': template['prefix'] + raw['instruction'] + template['suffix'],
            'label': raw['output']
        } for raw in tqdm(raw_data, desc="pre-processing data")
    ]
    return data
    