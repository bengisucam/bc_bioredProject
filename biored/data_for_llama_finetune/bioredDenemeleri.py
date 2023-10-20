from datasets import load_dataset
import pandas as pd

dataset = load_dataset("bigbio/biored", split="train")

train_data = pd.DataFrame(dataset)
print(train_data.head())
print(train_data.tail())

print(train_data.iloc[1])