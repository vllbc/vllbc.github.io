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
