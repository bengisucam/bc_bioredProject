import os

import nltk
import pandas as pd
from nltk.stem import PorterStemmer, WordNetLemmatizer
import numpy as np


def read_from_txt(file_path):
    f = open(file_path)
    with f as file:
        lines = [line.rstrip() for line in file]

    response_list = []
    for x in range(len(lines) - 6):
        if "### Response:" in lines[x]:
            if "INFO" in lines[x + 6]:
                response_list.append([lines[x + 1]])
    print(len(response_list))


def read_from_excel(file_path):
    df = pd.DataFrame(pd.read_excel(file_path))
    return df


def lemmatize_word(word):
    lemma = lemmatizer.lemmatize(word, "v")
    return lemma


# def calculate_lemmas_precision_recall(predicted_tokens, actual_tokens):
#     # Convert tokens to lowercase
#     predicted_tokens = [lemmatize_word(token.lower()) for token in predicted_tokens]
#     actual_tokens = [lemmatize_word(token.lower()) for token in actual_tokens]
#
#     TP = sum(1 for token in predicted_tokens if token in actual_tokens)
#     FP = sum(1 for token in predicted_tokens if token not in actual_tokens)
#     FN = sum(1 for token in actual_tokens if token not in predicted_tokens)
#
#     precision = TP / (TP + FP) if (TP + FP) > 0 else 0
#     recall = TP / (TP + FN) if (TP + FN) > 0 else 0
#
#     return precision, recall


def calculate_lemmas_metric(predicted_tokens, actual_tokens):
    # Convert tokens to lowercase
    predicted_tokens = [
        [lemmatize_word(token.lower()) if token is not None else "NoNe" for token in sublist]
        for sublist in predicted_tokens
    ]
    actual_tokens = [
        [lemmatize_word(token.lower()) for token in sublist]
        for sublist in actual_tokens
    ]

    predicted = [x[0].split(",") for x in predicted_tokens]
    actual = [x[0].split(",") for x in actual_tokens]

    scores = []
    for i in range(len(actual)):
        pred_sublist = predicted[i]
        actual_sublist = actual[i]

        TP = sum(1 for token in actual_sublist if token in pred_sublist)
        FP = 0
        FN = sum(1 for token in actual_sublist if token not in pred_sublist)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        scores.append({"precision": precision, "recall": recall, "f1": f1})

    return scores




if __name__ == '__main__':
    # pth_to_file = r"C:\Users\B3LAB\Desktop\thesis\Llama\llama2\inference_results\few-shot\lll_few_shot_inference_7B_chat_portion1.log"
    # read_from_txt(pth_to_file)

    pth_to_file = r"C:\Users\B3LAB\Desktop\thesis\Llama\llama2\inference_results\zero-shot\zero_shot_inference_70b_chat.xlsx"
    df = read_from_excel(pth_to_file)
    replaced_df = df.replace({np.nan: None})
    print(df["Predicted"])

    lemmatizer = WordNetLemmatizer()
    # stemmer = PorterStemmer()

    list_of_words = ["changing", "changed", "changes", "activated", "activates", "depends",
                     "dogs", "regulon", "interaction", "interacting", "controlled", "inhibits"]

    list_of_lemmas = [lemmatizer.lemmatize(i, "v") for i in list_of_words]
    # list_of_stems = [stemmer.stem(i) for i in list_of_words]
    # print(list_of_lemmas)

    predicted_tokens = [[value] for value in replaced_df["Predicted"]]
    actual_tokens = [[value] for value in replaced_df["True Label"]]
    scores_list = calculate_lemmas_metric(predicted_tokens, actual_tokens)
    for i in range(len(scores_list)):
        df.iloc[i]["Predicted"]
