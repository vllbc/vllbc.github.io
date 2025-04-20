# generate

理论部分在这：[generate相关](../../NLP/LLM/generate相关.md)
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
- **logits_processor**：高级功能，logits_processor 可以在每个 step 的输出概率计算完成后，对分数进行进一步的干预，改变输出的概率分布，从而影响生成的结果，例如最常见的，重复惩罚，就是使用 logits_processor 完成的。（不懂的话可以看后面如何具体实现的）
- **stopping_criteria**：高级功能，允许用户通过 stopping_criteria 自定义生成停止条件（不懂的话可以看后面如何具体实现的）
- prefix_allowed_tokens_fn：解码策略的一个超参数，用于前缀 token 约束（感觉没必要放在这里）
- synced_gpus：
- DeepSpeed ZeRO Stage-3 多GPU时使用（ZeRO-3包括优化器状态+梯度+权重并行优化，而推理阶段只使用权重并行），此时需要将 synced_gpus 设置成 Ture。.
- 否则，如果一个 GPU 在另一个 GPU 之前完成生成，整个系统就会挂起，因为其余 GPU 尚未从最先完成的 GPU 接收到权重分片。
- transformers>=4.28 在生成时检测到多个 GPU 会自动设置 synced_gpus=True，transformers<4.28 需要手动设置，本文代码环境transformers=4.41.1
- assistant_model：高级功能，辅助生成模型，另一个词表完全相同的小模型，有些token使用辅助模型生成更快
- streamer：流式输出控制器，现在的大模型平台都是一个字一个字显示出来的，这就是流式输出，否则的话会等所有生成完成再显示出来。这个可以自定义流式输出的方式
- negative_prompt_ids：负面提示，一些前沿研究会用到，不用管
- negative_prompt_attention_mask：负面提示的 attention_mask
- **kwargs
	- 以上输入都太高大上了，只有 inputs 会每次传入，其他的对于常规输出根本用不到（其实 inputs 也可以不用输入，通过tokenizer()得到model_inputs后，使用**model_inputs方式也可以传入）
	- 回想一下别人的代码，会看到这里经常传入 temperature=0.7, top_k=20, max_new_tokens=512等参数，都是通过**kwargs传入进来的
	- 其实传入的这些都是输入参数 generation_config 的属性（可以进入对应类中看一下有哪些属性，from transformers.generation.configuration_utils import GenerationConfig），你可以创建该对象并覆盖某些参数，也可以通过参数形式在调用.generate()时传进来
	- 在后面会将传入的这些参数覆盖掉generation_config中对应的属性

## inputs处理
```python
def _prepare_model_inputs(
    self,
    inputs: Optional[torch.Tensor] = None,
    bos_token_id: Optional[torch.Tensor] = None,
    model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
    """
    This function extracts the model-specific `inputs` for generation.
    """
    # 1.有一些 encoder-decoder 模型的输入有不同的名称，这里首先确认名称
    if (
        self.config.is_encoder_decoder
        and hasattr(self, "encoder")
        and self.encoder.main_input_name != self.main_input_name
    ):
        input_name = self.encoder.main_input_name
    else:
        input_name = self.main_input_name

    # 从 model_kwargs 中去掉 input_name: None 的键值对
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}

    # 2.这里确保 model.generate() 输入参数中的 inputs 和 kwargs 中的 input_name 只输入一个
    inputs_kwarg = model_kwargs.pop(input_name, None)
    if inputs_kwarg is not None and inputs is not None:
        raise ValueError(
            f"`inputs`: {inputs}` were passed alongside {input_name} which is not allowed. "
            f"Make sure to either pass {inputs} or {input_name}=..."
        )
    elif inputs_kwarg is not None:
        inputs = inputs_kwarg

    # 3.如果 input_name != inputs_embeds， 这里确保 input_name 和 inputs_embeds 只输入一个
    if input_name == "input_ids" and "inputs_embeds" in model_kwargs:
        # 如果是 decoder-only 模型，先看看模型 .forward() 函数的参数中，是否包含 inputs_embeds，如果不包含就弹出异常
        if not self.config.is_encoder_decoder:
            has_inputs_embeds_forwarding = "inputs_embeds" in set(
                inspect.signature(self.prepare_inputs_for_generation).parameters.keys()
            )
            if not has_inputs_embeds_forwarding:
                raise ValueError(
                    f"You passed `inputs_embeds` to `.generate()`, but the model class {self.__class__.__name__} "
                    "doesn't have its forwarding implemented. See the GPT2 implementation for an example "
                    "(https://github.com/huggingface/transformers/pull/21405), and feel free to open a PR with it!"
                )
            # In this case, `input_ids` is moved to the `model_kwargs`, so a few automations (like the creation of
            # the attention mask) can rely on the actual model input.
            model_kwargs["input_ids"] = self._maybe_initialize_input_ids_for_generation(
                inputs, bos_token_id, model_kwargs=model_kwargs
            )
        else:
            if inputs is not None:
                raise ValueError("You passed `inputs_embeds` and `input_ids` to `.generate()`. Please pick one.")
        inputs, input_name = model_kwargs["inputs_embeds"], "inputs_embeds"

    # 4. 如果 `inputs` 还是 None，尝试用 BOS token 创建 `input_ids`
    inputs = self._maybe_initialize_input_ids_for_generation(inputs, bos_token_id, model_kwargs)
    return inputs, input_name, model_kwargs

```
若传入了 inputs，就不要在 kwargs 中再次定义 input_ids
若 inputs 为 None，且 model_kwargs 不包含 input_ids 或 input_ids 也为 None，则创建一个 [batch_size, 1] 大小的tensor，里面的值都为 bos_token_id


## 参考
https://blog.csdn.net/qq_41496421/article/details/142346738?spm=1001.2014.3001.5502
https://blog.csdn.net/qq_41496421/article/details/142580960?spm=1001.2014.3001.5501
