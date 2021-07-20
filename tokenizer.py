from datasets import Dataset
from transformers import RobertaTokenizer, AutoTokenizer
import pandas as pd

print("Loading training data from disk...")
df_train = pd.read_pickle("data/python_train.pkl")
train = Dataset.from_pandas(df_train[["code", "docstring"]])
print("training data loaded.")
print("loading validaiton data from disk...")
df_valid = pd.read_pickle("data/python_valid.pkl")
valid = Dataset.from_pandas(df_valid[["code", "docstring"]])
print("validaiton data loaded.")

doc_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
code_tokenizer = AutoTokenizer.from_pretrained('huggingface/CodeBERTa-small-v1')

def tokenise_docstrings(element):
    return doc_tokenizer(element['docstring'], truncation=True)


def tokenise_code(element):
    return code_tokenizer(element['code'], truncation=True)


print("tokenising training data code...")
code_tokenised = train.map(tokenise_code, batched=True)
code_tokenised = code_tokenised.rename_column('attention_mask', 'encoder_attention_mask')
code_tokenised = code_tokenised.rename_column('input_ids', 'encoder_input_ids')
code_tokenised = code_tokenised.remove_columns('code')
print("tokenising training data docstrings...")
tokenised_dataset = code_tokenised.map(tokenise_docstrings, batched=True)
tokenised_dataset = tokenised_dataset.rename_column('attention_mask', 'decoder_attention_mask')
tokenised_dataset = tokenised_dataset.rename_column('input_ids', 'decoder_input_ids')
tokenised_dataset = tokenised_dataset.remove_columns('docstring')
print("training data tokenized.")
print("saving training data...")
tokenised_dataset.save_to_disk('data/python_tokenised/train')

print("Tokenizing validation data code...")
code_tokenised = valid.map(tokenise_code, batched=True)
code_tokenised = code_tokenised.rename_column('attention_mask', 'encoder_attention_mask')
code_tokenised = code_tokenised.rename_column('input_ids', 'encoder_input_ids')
code_tokenised = code_tokenised.remove_columns('code')
print("tokenising validation data docstrings...")
tokenised_dataset = code_tokenised.map(tokenise_docstrings, batched=True)
tokenised_dataset = tokenised_dataset.rename_column('attention_mask', 'decoder_attention_mask')
tokenised_dataset = tokenised_dataset.rename_column('input_ids', 'decoder_input_ids')
tokenised_dataset = tokenised_dataset.remove_columns('docstring')
print("validation data tokenized.")

print("saving validation data...")
tokenised_dataset.save_to_disk('data/python_tokenised/val')
print("Validation data tokenized.")
xx = 0