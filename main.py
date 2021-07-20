import torch
import pandas as pd
from data.dataloaders import SemanticSearchDataCollator
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer, AutoModelForMaskedLM, RobertaTokenizer, RobertaForCausalLM, RobertaConfig
from transformers import EncoderDecoderModel, AutoConfig, EncoderDecoderConfig, Seq2SeqTrainingArguments, Trainer, DataCollatorForSeq2Seq, Seq2SeqTrainer

# define an encoder model to represenet the code
code_config = AutoConfig.from_pretrained("huggingface/CodeBERTa-small-v1")
code_encoder = AutoModelForMaskedLM.from_pretrained(
    "huggingface/CodeBERTa-small-v1",
    config=code_config
)

# and a decoder model to generate descriptions
doc_config = RobertaConfig.from_pretrained('roberta-base')
doc_config.is_decoder = True
# doc_config.add_cross_attention = True
doc_decoder = RobertaForCausalLM.from_pretrained(
    'roberta-base',
    config=doc_config
)

# combine these models into an encoder decoder, translating from code into natural language
config = EncoderDecoderConfig.from_encoder_decoder_configs(
    code_config,
    doc_config
)
model = EncoderDecoderModel(config=config)

# define a dataset object to hold data for the trainer
train = load_from_disk("data/python_tokenised/train")
val = load_from_disk("data/python_tokenised/val")

# todo resave a version of the datasets with these columns renamed and delete the following four lines
train = train.rename_column("encoder_input_ids", "input_ids")
train = train.rename_column("encoder_attention_mask", "attention_mask")
val = val.rename_column("encoder_input_ids", "input_ids")
val = val.rename_column("encoder_attention_mask", "attention_mask")

dataset = DatasetDict(
    {
        'train': train,
        'val': val
    }
)

# set the format of the dataset for a pytorch model
dataset['train'].set_format(
    type='torch',
    columns=[
        'input_ids',
        'attention_mask',
        'decoder_input_ids',
        'decoder_attention_mask'
    ]
)
dataset['val'].set_format(
    type='torch',
    columns=[
        'input_ids',
        'attention_mask',
        'decoder_input_ids',
        'decoder_attention_mask'
    ]
)

# define tokenizers for the encoder and decoder
doc_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
code_tokenizer = AutoTokenizer.from_pretrained('huggingface/CodeBERTa-small-v1')

# define a dataloader and collating function
# train_dataloader = torch.utils.data.DataLoader(dataset['train'], batch_size=32)
data_collator = SemanticSearchDataCollator(code_tokenizer, doc_tokenizer)

# define a class to hold model parameters for training
training_args = Seq2SeqTrainingArguments(
    output_dir="v1-code2tag",
    evaluation_strategy="epoch",
    num_train_epochs=5,
    per_device_eval_batch_size=8,
    per_device_train_batch_size=8
)

# define a trainer to run training
trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['val'],
    data_collator=data_collator
)

# run training and save the model
trainer.train(resume_from_checkpoint="v1-code2tag/checkpoint-36000")
trainer.save_model("models")

