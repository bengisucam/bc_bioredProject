
import pandas as pd
import json
from datasets import load_dataset
from peft import LoraConfig
import os

def load_and_test(dataset, split):
    ds = load_dataset(dataset)
    return ds.head()

if __name__ == '__main__':
    print(os.getcwd())
    train_df = pd.read_csv("./out_files_masked/LLL-train.csv")
    # val_df = pd.read_csv("LLL_with_INO/out_files_protein_tagged_byhand/LLL-val.csv")
    test_df = pd.read_csv("./out_files_masked/LLL-test.csv")

    all_data = {
    "train" : train_df.to_dict(orient="records"),
    "test" : test_df.to_dict(orient="records"),
    #"val" : val_df.to_dict(orient="records")
    }

    for split in all_data:
        f_name = "./out_files_masked_hf/" + split + ".jsonl"
        with open(f_name, "w") as f:
            for line in all_data[split]:
                f.write(json.dumps(line) + "\n")
        f.close()

    # test_head = load_and_test("bengisucam/HPRD50_only_true", split="test")
    # print(test_head)
