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


def apply_lemma_normalization(lemma_list):

    for i in range(len(lemma_list)):
        if lemma_list[i] == "inducible":
            lemma_list[i] = "induce"
        elif lemma_list[i] == "gene expression":
            lemma_list[i] = "expression"

        elif lemma_list[i] == "dependent genes":
            lemma_list[i] = "dependent"
        elif lemma_list[i] == "indirectly activate":
            lemma_list[i] = "activate"
        elif lemma_list[i] == "promoter elements":
            lemma_list[i] = "promoter"
        elif lemma_list[i] == "under control":
            lemma_list[i] = "control"
        elif lemma_list[i] == "under the control":
            lemma_list[i] = "control"
        elif lemma_list[i] == "under control of":
            lemma_list[i] = "control"
        elif lemma_list[i] == "member":
            lemma_list[i] = "member of"
        elif lemma_list[i] == "regulate":
            lemma_list[i] = "regulon"
        elif lemma_list[i] == "negatively regulate":
            lemma_list[i] = "regulon"
        elif lemma_list[i] == "negative regulate":
            lemma_list[i] = "negatively regulate"
        elif lemma_list[i] == "driven by":
            lemma_list[i] = "drive"
        elif lemma_list[i] == "produce":
            lemma_list[i] = "production"
        elif lemma_list[i] == "dependence":
            lemma_list[i] = "depend"
        elif lemma_list[i] == "activation":
            lemma_list[i] = "activate"
        elif lemma_list[i] == "combined action":
            lemma_list[i] = "action"
        elif lemma_list[i] == "inducer":
            lemma_list[i] = "induce"
        elif lemma_list[i] == "induction":
            lemma_list[i] = "induce"
        elif lemma_list[i] == "regulons":
            lemma_list[i] = "regulon"

    return lemma_list


def calculate_lemmas_metric(predicted, actual, is_normalized):
    scores = []
    for i in range(len(actual)):
        if is_normalized:
            pred_sublist = apply_lemma_normalization(predicted[i])
            actual_sublist = apply_lemma_normalization(actual[i])
        else:
            pred_sublist = predicted[i]
            actual_sublist = actual[i]

        TP = sum(1 for token in actual_sublist if token in pred_sublist)
        FP = sum(1 for token in pred_sublist if token not in actual_sublist)
        FN = sum(1 for token in actual_sublist if token not in pred_sublist)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        scores.append({"precision": precision, "recall": recall, "f1": f1})

    return scores




if __name__ == '__main__':


    # is_normalized = False
    # #pth_to_file = r"C:\Users\B3LAB\Desktop\thesis\Llama\llama2\inference_results\zero-shot\zero_shot_inference_70b_chat.xlsx"
    # #pth_to_file = r"C:\Users\B3LAB\Desktop\thesis\Llama\llama2\finetune_results\finetuned_13b_chat_7.xlsx"
    # #pth_to_file = r"C:\Users\B3LAB\Desktop\thesis\Llama\llama2\inference_results\few-shot\few_shot_inference_70b_chat_P3.xlsx"
    # pth_to_file = r"C:\Users\B3LAB\Desktop\thesis\Llama\llama2\finetune_results\kfold\finetuned_7b_chat_7_5fold_5.xlsx"
    # df = read_from_excel(pth_to_file)
    # replaced_df = df.replace({np.nan: None})

    lemmatizer = WordNetLemmatizer()
    # stemmer = PorterStemmer()

    list_of_words = ["changing", "changed", "changes", "activated", "activates", "depends", "depended", "dependent", "dependence",
                     "dogs", "regulon", "interaction", "interacting", "controlled", "inhibits", "drives",
                     "negative regulation", "indirectly activate", "promoter elements", "driven",
                     "produced", "production", "regulon", "regulated", "induction", "regulons"]

    list_of_lemmas = [lemmatizer.lemmatize(i, "v") for i in list_of_words]
    #list_of_stems = [stemmer.stem(i) for i in list_of_words]
    print(list_of_lemmas)
    #
    # predicted_tokens = [[value] for value in replaced_df["Predicted"]]
    # actual_tokens = [[value] for value in replaced_df["True Label"]]
    #
    # predicted_lemmas, actual_lemmas = get_lemma(predicted_tokens, actual_tokens)
    #
    # scores_list = calculate_lemmas_metric(predicted_lemmas, actual_lemmas, is_normalized=is_normalized)
    #
    # pred_lemma_column = [predicted_lemmas[i] for i in range(len(scores_list))]
    # act_lemma_column = [actual_lemmas[i] for i in range(len(scores_list))]
    # precision_column = [scores_list[i]["precision"] for i in range(len(scores_list))]
    # recall_column = [scores_list[i]["recall"] for i in range(len(scores_list))]
    # f1_column = [scores_list[i]["f1"] for i in range(len(scores_list))]
    #
    # df["Predicted Lemma"] = pred_lemma_column
    # df["True Lemma"] = act_lemma_column
    # df["Precision"] = precision_column
    # df["Recall"] = recall_column
    # df["F1"] = f1_column
    #
    # ave_precision = df["Precision"].mean()
    # ave_recall = df["Recall"].mean()
    # ave_f1 = df["F1"].mean()
    #
    # df.loc[-1] = ["Average Scores", None, None, None, None, ave_precision, ave_recall, ave_f1]
    #
    # print("Average Precision: ", ave_precision)
    # print("Average Recall: ", ave_recall)
    # print("Average F1: ", ave_f1)
    #
    #
    # #pth_to_output_file = r"C:\Users\B3LAB\Desktop\thesis\Llama\llama2\inference_results\zero-shot\lemma_metrics\metric-70b-chat.xlsx"
    # #pth_to_output_file = r"C:\Users\B3LAB\Desktop\thesis\Llama\llama2\finetune_results\lemma_metrics\metric-13b-chat-7.xlsx"
    # pth_to_output_file = r"C:\Users\B3LAB\Desktop\thesis\Llama\llama2\finetune_results\kfold\lemma_metrics\metric-7b-chat-7-5fold5.xlsx"
    # df.to_excel(pth_to_output_file)
