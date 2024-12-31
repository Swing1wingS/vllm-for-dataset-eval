
echo "=========================== Predicting ==========================="
exp="main-$1"
tmp_train_dir="/path/to/lora_checkpoint"
eval_dir="results/eval-${exp}"
mkdir -p $eval_dir
# cmd="CUDA_VISIBLE_DEVICES=2,3 python ${exp}.py \
cmd="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ${exp}.py \
    --seed 42 \
    --model_name_or_path pretrained_models/Meta-Llama-3.1-8B-Instruct \
    --template llama3 \
    --dataset_dir data \
    --eval_dataset dataset_name \
    --cutoff_len 2400 \
    --max_new_tokens 24 \
    --top_p 0.7 \
    --temperature 0.95 \
    --lora_adapter_path ${tmp_train_dir} \
    --output_dir ${eval_dir} | tee ${eval_dir}/${exp}.log"

echo $cmd
eval $cmd
