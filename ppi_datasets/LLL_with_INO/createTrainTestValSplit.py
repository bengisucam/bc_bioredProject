import pandas as pd
import numpy as np
import os

print(os.getcwd())


def split_train_test(df):
    num_of_rows = df.index.stop
    all_indices = [i for i in range(num_of_rows)]
    test = df.sample(frac=0.15, random_state=42)
    test_indices = np.ravel(test.index).tolist()
    train_indices = [i for i in all_indices if i not in test_indices]

    train = df[df.index.isin(train_indices)]
    val = train.sample(frac=0.15, random_state=44)
    val_indices = np.ravel(val.index).tolist()

    # train.reset_index(inplace=True)
    # test.reset_index(inplace=True)
    return train, test, val


if __name__ == '__main__':
    df_all = pd.read_excel('LLL.xlsx')
    train_df, test_df, val_df = split_train_test(df_all)
    print(os.getcwd())
    train_df.to_csv(os.getcwd() + "\\out_files\\LLL-train.csv", encoding='utf-8')
    test_df.to_csv(os.getcwd() + "\\out_files\\LLL-test.csv", encoding='utf-8')
    val_df.to_csv(os.getcwd() + "\\out_files\\LLL-val.csv", encoding='utf-8')

