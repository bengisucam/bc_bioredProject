import pandas as pd
import numpy as np
import os


if __name__ == '__main__':
    train_df = pd.read_csv("out_files/LLL-train.csv")
    val_df = pd.read_csv("out_files/LLL-val.csv")
    test_df = pd.read_csv("out_files/LLL-test.csv")
    train_val_dict = {"train": train_df, "val": val_df}

    for split in train_val_dict.keys():
        prompts = []
        for i in list(train_val_dict[split].index.array):
            row = train_df.iloc[i]
            sentence = row["Sentence"]
            keywords = row["Keywords"]
            question = "What is the keyword that represents the  interaction between the proteins " + row["Gene1"] + " and " + row["Gene2"] + " in the given sentence?"
            prompt = question + " " + sentence + " " + "Keywords: " + keywords
            prompts.append(prompt)
        train_val_dict[split]['Prompts'] = prompts
        train_val_dict[split].to_csv(os.getcwd() + "\\prompt_files\\LLL-" + split + ".csv", encoding='utf-8')

    prompts = []
    for i in list(test_df.index.array):
        sentence = test_df.iloc[i]["Sentence"]
        keywords = test_df.iloc[i]["Keywords"]
        question = "What is the key word that represents the  interaction between the proteins " + train_df.iloc[i][
            "Gene1"] + " and " + train_df.iloc[i]["Gene2"] + " in the given sentence?"
        prompt = question + " " + sentence + " " + "Keywords: "
        prompts.append(prompt)
    test_df['Prompts'] = prompts
    test_df.to_csv(os.getcwd() + "\\prompt_files\\LLL-test.csv", encoding='utf-8')