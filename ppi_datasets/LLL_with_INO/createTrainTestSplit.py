import pandas as pd
import numpy as np
import os

print(os.getcwd())


def split_train_test(df):
    num_of_rows = df.index.stop
    all_indeces = [i for i in range(num_of_rows)]
    test = df.sample(frac=0.2, random_state=42)
    test_indeces = np.ravel(test.index).tolist()
    train_indeces = [i for i in all_indeces if i not in test_indeces]
    train = df[df.index.isin(train_indeces)]
    # train.reset_index(inplace=True)
    # test.reset_index(inplace=True)
    return test, train


if __name__ == '__main__':
    df_all = pd.read_excel('LLL.xlsx')
    test_df, train_df = split_train_test(df_all)
    print(os.getcwd())
    train_df.to_csv(os.getcwd() + "\\out_files\\LLL-train.csv", encoding='utf-8')
    test_df.to_csv(os.getcwd() + "\\out_files\\LLL-test.csv", encoding='utf-8')

