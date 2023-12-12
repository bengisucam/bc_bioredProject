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
    lemmatizer = WordNetLemmatizer()
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

def get_lemma(predicted_tokens, actual_tokens):
    # Convert tokens to lowercase
    predicted_tokens = [
        [token.lower() if token is not None else "NoNe" for token in sublist]
        for sublist in predicted_tokens
    ]
    actual_tokens = [
        [token.lower() for token in sublist]
        for sublist in actual_tokens
    ]

    predicted = []
    actual = []

    for x in actual_tokens:
        values = []
        for y in [x[0].split(",")]:
            if len(y) > 1:
                for z in y:
                    values.append(lemmatize_word(z.strip()))
            else:
                values.append(lemmatize_word(y[0]))
        actual.append(values)

    for x in predicted_tokens:
        values = []
        for y in [x[0].split(",")]:
            if len(y) > 1:
                for z in y:
                    values.append(lemmatize_word(z.strip()))
            else:
                values.append(lemmatize_word(y[0]))
        predicted.append(values)



    return predicted, actual


def calculate_lemmas_metric(predicted, actual):
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

    #pth_to_file = r"C:\Users\B3LAB\Desktop\thesis\Llama\llama2\inference_results\zero-shot\zero_shot_inference_13b_chat.xlsx"
    #pth_to_file = r"C:\Users\B3LAB\Desktop\thesis\Llama\llama2\finetune_results\finetuned_13b_chat_4.xlsx"
    pth_to_file = r"C:\Users\B3LAB\Desktop\thesis\Llama\llama2\inference_results\few-shot\few_shot_inference_70b_chat_P3.xlsx"
    df = read_from_excel(pth_to_file)
    replaced_df = df.replace({np.nan: None})

    lemmatizer = WordNetLemmatizer()
    # stemmer = PorterStemmer()

    list_of_words = ["changing", "changed", "changes", "activated", "activates", "depends",
                     "dogs", "regulon", "interaction", "interacting", "controlled", "inhibits", "drives"]

    list_of_lemmas = [lemmatizer.lemmatize(i, "v") for i in list_of_words]
    #list_of_stems = [stemmer.stem(i) for i in list_of_words]
    print(list_of_lemmas)

    predicted_tokens = [[value] for value in replaced_df["Predicted"]]
    actual_tokens = [[value] for value in replaced_df["True Label"]]

    predicted_lemmas, actual_lemmas = get_lemma(predicted_tokens, actual_tokens)

    scores_list = calculate_lemmas_metric(predicted_lemmas, actual_lemmas)

    pred_lemma_column = [predicted_lemmas[i] for i in range(len(scores_list))]
    act_lemma_column = [actual_lemmas[i] for i in range(len(scores_list))]
    precision_column = [scores_list[i]["precision"] for i in range(len(scores_list))]
    recall_column = [scores_list[i]["recall"] for i in range(len(scores_list))]
    f1_column = [scores_list[i]["f1"] for i in range(len(scores_list))]

    df["Predicted Lemma"] = pred_lemma_column
    df["True Lemma"] = act_lemma_column
    df["Precision"] = precision_column
    df["Recall"] = recall_column
    df["F1"] = f1_column

    ave_precision = df["Precision"].mean()
    ave_recall = df["Recall"].mean()
    ave_f1 = df["F1"].mean()

    df.loc[-1] = ["Average Scores", None, None, None, None, ave_precision, ave_recall, ave_f1]

    print("Average Precision: ", ave_precision)
    print("Average Recall: ", ave_recall)
    print("Average F1: ", ave_f1)


    #pth_to_output_file = r"C:\Users\B3LAB\Desktop\thesis\Llama\llama2\inference_results\zero-shot\metric-13b-chat.xlsx"
    #pth_to_output_file = r"C:\Users\B3LAB\Desktop\thesis\Llama\llama2\finetune_results\metric-13b-chat-4.xlsx"
    pth_to_output_file = r"C:\Users\B3LAB\Desktop\thesis\Llama\llama2\inference_results\few-shot\metric-70b-chat-P3.xlsx"
    df.to_excel(pth_to_output_file)
