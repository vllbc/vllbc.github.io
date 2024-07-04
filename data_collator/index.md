# 情感分析

目的：**手动将抽取出的样本堆叠起来，构造成batch**
Trainer函数有一个参数data_collator，其值也为一个函数，用于从一个list of elements来构造一个batch。
如下
```python
trainer = CLTrainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset if training_args.do_train else None,

        tokenizer=tokenizer,

        data_collator=data_collator, 

    )
```

其中data_collator为自定义的类，必须可调用，因此实现了__call__，要实现输入额外参数，就要用类的方式或者再嵌套匿名函数实现。

```python
@dataclass

    class OurDataCollatorWithPadding:

  

        tokenizer: PreTrainedTokenizerBase

        padding: Union[bool, str, PaddingStrategy] = True

        max_length: Optional[int] = None

        pad_to_multiple_of: Optional[int] = None

        mlm: bool = True

        mlm_probability: float = data_args.mlm_probability

  

        def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

            special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels'] # 必有的输入参数

            bs = len(features)

            if bs > 0:

                num_sent = len(features[0]['input_ids'])

            else:

                return

            flat_features = []

            for feature in features:

                for i in range(num_sent):

                    flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

  

            batch = self.tokenizer.pad(

                flat_features,

                padding=self.padding,

                max_length=self.max_length,

                pad_to_multiple_of=self.pad_to_multiple_of,

                return_tensors="pt",

            )

            if model_args.do_mlm:

                batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])

  

            batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}

  

            if "label" in batch:

                batch["labels"] = batch["label"]

                del batch["label"]

            if "label_ids" in batch:

                batch["labels"] = batch["label_ids"]

                del batch["label_ids"]

  

            return batch
```

并且返回的必须是dict类型，必须要有特定的参数，即model.forward的必要参数。

而dataloader中的collate_fn比较自由，返回任意形式都可以。
collate_fn的用处:
- 自定义数据堆叠过程
- 自定义batch数据的输出形式
