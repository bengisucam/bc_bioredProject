import pandas as pd
import numpy as np
import os


if __name__ == '__main__':
    train_df = pd.read_csv("out_files/LLL-train.csv")
    test_df = pd.read_csv("out_files/LLL-test.csv")

    prompts = []
    for i in list(train_df.index.array):
        row = train_df.iloc[i]
        sentence =row["Sentence"]
        keywords = row["Keywords"]
        question = "What is the keyword that represents the  interaction between the proteins " + row["Gene1"] + " and " + row["Gene2"] + " in the given sentence?"
        prompt = question + " " + sentence + " " + "Keywords: " + keywords
        prompts.append(prompt)
    train_df['Prompts'] = prompts

    prompts = []
    for i in list(test_df.index.array):
        sentence = test_df.iloc[i]["Sentence"]
        keywords = test_df.iloc[i]["Keywords"]
        question = "What is the key word that represents the  interaction between the proteins " + train_df.iloc[i][
            "Gene1"] + " and " + train_df.iloc[i]["Gene2"] + " in the given sentence?"
        prompt = question + sentence + "Keywords: "
        prompts.append(prompt)
    test_df['Prompts'] = prompts

    train_df.to_csv(os.getcwd() + "\\prompt_files\\LLL-train.csv", encoding='utf-8')
    test_df.to_csv(os.getcwd() + "\\prompt_files\\LLL-test.csv", encoding='utf-8')