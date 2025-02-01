from transformers import GPT2Tokenizer
from datasets import load_dataset

# Loading the dataset to train our model
class loadDataset:
      
   dataset = load_dataset("mteb/tweet_sentiment_extraction")
   def tokenize_function(examples):
      tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
      tokenizer.pad_token = tokenizer.eos_token
      return tokenizer(examples["text"], padding="max_length", truncation=True)

   tokenized_datasets = dataset.map(tokenize_function, batched=True)
