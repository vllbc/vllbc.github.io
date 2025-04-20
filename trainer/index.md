# trainer

# 基本用法

下面是使用的一个例子，重点是TrainingArg和data_collator。

```python
dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path='./text.txt', block_size=512)

data_collator = DataCollatorForLanguageModeling( tokenizer=tokenizer, mlm=True, mlm_probability=0.15 ) 

training_args = TrainingArguments( output_dir='./outputs/',
								  overwrite_output_dir=True, 
								  num_train_epochs=100, 
								  per_device_train_batch_size=16, 
								  save_steps=5000, ) 

trainer = Trainer( model=model, 
				  args=training_args, 
				  data_collator=data_collator, 
				  train_dataset=dataset, ) 

trainer.train() 

trainer.save_model('./outputs/')


```

的附图作简单地介绍，显而易见地，下面描述中的附图仅仅是本发明的一些实施例，对于本领域普通技术人员来讲，在不付出创造性劳动的前提下，还可以根据这些附图获得其它的附图。
[0089] 图1为本发明第一个实施例提供的一种基于大语言模型自身对上下文进行压缩的
方法的整体流程图；
[0090] 图2为本发明第一个实施例提供的一种基于大语言模型自身对上下文进行压缩的
方法的Decoder-only模型架构示意图；
[0091] 图3为本发明第一个实施例提供的一种基于大语言模型自身对上下文进行压缩的
方法的利用现有大语言模型训练流程图；
[0092] 图4为本发明第一个实施例提供的一种基于大语言模型自身对上下文进行压缩的
方法的预训练大语言模型流程图；
[0093] 图5为本发明第一个实施例提供的一种基于大语言模型自身对上下文进行压缩的
方法的推理流程图；
[0094] 图6为本发明第一个实施例提供的一种基于大语言模型自身对上下文进行压缩的
方法的虚拟字符检索流程图；
[0095] 图7为本发明第二个实施例提供的一种基于大语言模型自身对上下文进行压缩的
方法的各个模型的推理性能对比图；
[0096] 图8为本发明第二个实施例提供的一种基于大语言模型自身对上下文进行压缩的
方法的部分压缩示例图。

### 具体实施方式

[0097] 为使本发明的上述目的、特征和优点能够更加明显易懂，下面结合说明书附图对本发明的具体实施方式做详细的说明，显然所描述的实施例是本发明的一部分实施例，而不是全部实施例。基于本发明中的实施例，本领域普通人员在没有做出创造性劳动前提下所获得的所有其他实施例，都应当属于本发明的保护的范围。
[0098] 实施例1
[0099] 参照图1~图6，为本发明的一个实施例，提供了一种基于大语言模型自身对上下文
进行压缩的方法，包括：
[0100] Sl:获取待压缩文本，添加任务描述、分隔符和压缩槽。
[0101] 添加任务描述、分隔符和压缩槽包括，将任务描述、待压缩文本和连续掩码序列拼

接成一个新的序列 sequence:

AN

$ESTAC$

$sequence=(x_{p},x_{c},x_{m})$

d$x= \frac 12$ d$x= \frac 12$ d$x= \frac 12$

[0102]

$$x_m=[M][M]\cdotp\cdotp\cdotp[M]$$

[0103] 其中，$x_p$表示任务描述，$x_\mathrm{c}$表示待压缩文本，$[M]$表示压缩槽，$x_m$表示连
续掩码序列。
[0104] 应说明的是，任务描述来源于预设的数据库。

