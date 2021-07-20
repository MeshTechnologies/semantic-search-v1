from datasets import Dataset
from transformers import RobertaTokenizer, AutoTokenizer
import pandas as pd

print("Loading training data from disk...")
df_train = pd.read_pickle("data/python/hf_val_doc/python_train.pkl")
train = Dataset.from_pandas(df_train[["code", "docstring"]])
print("training data loaded.")
print("loading validaiton data from disk...")
df_valid = pd.read_pickle("data/python/hf_val_doc/python_valid.pkl")
valid = Dataset.from_pandas(df_train[["code", "docstring"]])
print("validaiton data loaded.")
print("tokenising training data...")

doc_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
code_tokenizer = AutoTokenizer.from_pretrained('huggingface/CodeBERTa-small-v1')

def tokenise_docstrings(element):
    return doc_tokenizer(element['docstring'], truncation=True)


def tokenise_code(element):
    return code_tokenizer(element['code'], truncation=True)

code_tokenised = train.map(tokenise_code, batched=True)
code_tokenised.rename_column_('attention_mask', 'encoder_attention_mask')
code_tokenised.rename_column_('input_ids', 'encoder_input_ids')
code_tokenised.remove_columns_('code')
tokenised_dataset = code_tokenised.map(tokenise_docstrings, batched=True)

# checkpoint = "distilbert-base-uncased"
# tokenizer = DistilBertTokenizer.from_pretrained(checkpoint)
# train_tokenized = tokenizer(train["docstring"])
# print("training data tokenized.")
# print("tokenizing validation data....")
# valid_tokenized = tokenizer(valid["docstring"])
print("validation data tokenized.")
xx = 0