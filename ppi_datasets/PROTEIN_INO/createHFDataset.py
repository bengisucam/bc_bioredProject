
import pandas as pd
import json
from datasets import load_dataset
from peft import LoraConfig

def load_and_test(dataset, split):
    ds = load_dataset(dataset)
    return ds.head()

if __name__ == '__main__':
    # train_df = pd.read_csv("LLL_with_INO/out_files_protein_tagged_byhand/LLL-train.csv")
    # val_df = pd.read_csv("LLL_with_INO/out_files_protein_tagged_byhand/LLL-val.csv")
    test_df = pd.read_csv("HPRD50/llama2-70b-predicted-ino/HPRD50-test-tagged.csv")

    all_data = {
    # "train" : train_df.to_dict(orient="records"),
    "test" : test_df.to_dict(orient="records"),
    #"val" : val_df.to_dict(orient="records")
    }

    for split in all_data:
        f_name = "HPRD50/llama2-70b-predicted-ino/out_hf_files/" + split + ".jsonl"
        with open(f_name, "w") as f:
            for line in all_data[split]:
                f.write(json.dumps(line) + "\n")
        f.close()

    # test_head = load_and_test("bengisucam/HPRD50_only_true", split="test")
    # print(test_head)
