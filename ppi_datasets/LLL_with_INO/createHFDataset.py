
import pandas as pd
import json
from datasets import load_dataset

def load_and_test(dataset, split):
    ds = load_dataset(dataset)
    return ds.head()

if __name__ == '__main__':
    train_df = pd.read_csv("out_files/LLL-train.csv")
    val_df = pd.read_csv("out_files/LLL-val.csv")
    test_df = pd.read_csv("out_files/LLL-test.csv")

    all_data = {
    "train" : train_df.to_dict(orient="records"),
    "test" : test_df.to_dict(orient="records"),
    "val" : val_df.to_dict(orient="records")
    }

    for split in all_data:
        f_name = "out_hf_files/" + split + ".jsonl"
        with open(f_name, "w") as f:
            for line in all_data[split]:
                f.write(json.dumps(line) + "\n")
        f.close()

    test_head = load_and_test("bengisucam/LLL_INO", split="test")
    print(test_head)
