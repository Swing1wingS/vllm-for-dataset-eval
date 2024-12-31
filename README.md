# vllm-for-dataset-eval

## Motivation
[vllm](https://github.com/vllm-project/vllm)支持将整个数据集的prompt作为一个`list`参数传入`generate`方法，做[Batched Inference](https://docs.vllm.ai/en/stable/getting_started/quickstart.html#offline-batched-inference)，like this：（记为`sp-bi`，即`single process batched inference`）
```python
prompts = list(map(lambda x: x['llm_prompt'], data))
outputs = llm.generate(
    prompts, 
    sampling_params=sampling_params, 
    use_tqdm=True, 
    lora_request=LoRARequest('lora', 1, args.lora_adapter_path) if args.lora_adapter_path else None
)
for entry, out in zip(data, outputs):
    preds.append({
        "prompt": entry['llm_prompt'], 
        "predict": out.outputs[0].text,
        "label": entry['label']
    })
```
这样做虽然`vllm`内部可能有做优化，但是效率仍然不够高。
并且，如果对于某条数据想做多步推理，即输入多次，这个方法就完全做不了了。

## 思路

1. 可以考虑使用多进程对数据集作并行化。记为`mp-si`，即`multi process single inference`
    方法很简单，根据可用卡数`n`将数据集分为`n`份，每张卡分一个进程处理对应的数据集，每个进程独立执行，最终将返回结果收集起来，完全等价于`sp-bi`。
    此法还可支持**单条数据多步推理**。

2. 可以考虑结合`mp-si`和`sp-bi`，即仍然采用`mp-si`的多进程构建方案，但是每个进程内，对数据集作batched inference。记为`mp-bi`。


## 实验

|       | 推理时间 / s    | 任务性能 (F1)  |
|-------|----------------|-----------|
| sp-bi | 7588.8105     | 36.00     |
| mp-si | 945.0917      | 36.15     |
| mp-bi | **676.7724s** | **36.69** |

> 时间测量方式比较粗糙，仅供参考。

本实验在8张NVIDIA-A800-SXM4-80GB进行，使用Llama3.1-8B-Instruct+Lora进行推理，3种的setting采用了同一个评估数据集。
`mp-bi`的setting起到了很强的加速效果，同时任务性能与其他setting区别不大。
