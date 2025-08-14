# generate

理论部分在这：[generate相关](../../LLM/inference/generate相关.md)
## generate参数
```python
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

```
在代码中可以看到在函数入口显式的定义了很多参数。他们的具体含义如下

- inputs：tensor 形式的 token_id，通常先准备文本形式的提示词和输入，使用tokenizer转化为对应 id，这里维度通常为 [batch_size, seq_len]
- generation_config：一个用 GenerationConfig 类创建的对象，存储着模型生成的超参数，可以提前创建该对象并传入 .generate()
- **logits_processor**：高级功能，logits_processor 可以在每个 step 的输出概率计算完成后，对分数进行进一步的干预，改变输出的概率分布，从而影响生成的结果，例如最常见的，重复惩罚，就是使用 logits_processor 完成的。
- **stopping_criteria**：高级功能，允许用户通过 stopping_criteria 自定义生成停止条件
- prefix_allowed_tokens_fn：解码策略的一个超参数，用于前缀 token 约束
- synced_gpus：
- DeepSpeed ZeRO Stage-3 多GPU时使用（ZeRO-3包括优化器状态+梯度+权重并行优化，而推理阶段只使用权重并行），此时需要将 synced_gpus 设置成 Ture。.
- 否则，如果一个 GPU 在另一个 GPU 之前完成生成，整个系统就会挂起，因为其余 GPU 尚未从最先完成的 GPU 接收到权重分片。
- transformers>=4.28 在生成时检测到多个 GPU 会自动设置 synced_gpus=True，transformers<4.28 需要手动设置，本文代码环境transformers=4.41.1
- assistant_model：高级功能，辅助生成模型，另一个词表完全相同的小模型，有些token使用辅助模型生成更快
- streamer：流式输出控制器，现在的大模型平台都是一个字一个字显示出来的，这就是流式输出，否则的话会等所有生成完成再显示出来。这个可以自定义流式输出的方式
- negative_prompt_ids：负面提示，一些前沿研究会用到，不用管
- negative_prompt_attention_mask：负面提示的 attention_mask
- **kwargs
	- 这里经常传入 temperature=0.7, top_k=20, max_new_tokens=512等参数，都是通过**kwargs传入进来的
	- 其实传入的这些都是输入参数 generation_config 的属性（可以进入对应类中看一下有哪些属性，from transformers.generation.configuration_utils import GenerationConfig），你可以创建该对象并覆盖某些参数，也可以通过参数形式在调用.generate()时传进来
	- 在后面会将传入的这些参数覆盖掉generation_config中对应的属性

下面只说明一些关键的地方
## kwargs -> generation_config

就是将kwargs中传入的kwargs的参数变成config。
```
generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)

```
## 准备logit处理器

```python

prepared_logits_processor = self._get_logits_processor(
    generation_config=generation_config,
    input_ids_seq_length=input_ids_length,
    encoder_input_ids=inputs_tensor,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    logits_processor=logits_processor,
    device=inputs_tensor.device,
    model_kwargs=model_kwargs,
    negative_prompt_ids=negative_prompt_ids,
    negative_prompt_attention_mask=negative_prompt_attention_mask,
)

```
就是将generation_config中的采样参数封装成logit-processor，还有自己定义的processor

## 准备stopping处理器
```python

prepared_stopping_criteria = self._get_stopping_criteria(
    generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
)

```

同理。将一些与停止有关的参数封装成stopping处理器。

## logits warper

- `logits warper` 里面是采样时才需要运行的处理器
- `logits processor` 是通用的处理器，每种生成模式都需要用到的
```python

prepared_logits_warper = (
    self._get_logits_warper(generation_config) if generation_config.do_sample else None
)

```

## 正式生成

```python
# 进入模型内部生成下一个token
outputs = self(
    **model_inputs,
    return_dict=True,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
)
	
if synced_gpus and this_peer_finished:
    continue  # don't waste resources running the code we don't need

# 取出最后一个token，.logits维度为（batch_size, seq_len, vocab_size）
next_token_logits = outputs.logits[:, -1, :]

# 经过前面的处理器进行分数调整
next_token_scores = logits_processor(input_ids, next_token_logits)
if do_sample:
    next_token_scores = logits_warper(input_ids, next_token_scores)

```

按照是否采样来生成下一个token：
```python
if do_sample:
    probs = nn.functional.softmax(next_token_scores, dim=-1)
    # torch.multinomial：按照输入probs的每一行（每个batch）作为采样的概率，
    # 每行不放回的取出num_samples个，随机采样每个batch按输入概率取出一个
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
else:
	# torch.argmax取出输入next_token_scores中值最大的索引
    next_tokens = torch.argmax(next_token_scores, dim=-1)

```

最后判断是否可以停止：

```python

unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
this_peer_finished = unfinished_sequences.max() == 0

```
## 参考
https://blog.csdn.net/qq_41496421/article/details/142346738?spm=1001.2014.3001.5502
https://blog.csdn.net/qq_41496421/article/details/142580960?spm=1001.2014.3001.5501
