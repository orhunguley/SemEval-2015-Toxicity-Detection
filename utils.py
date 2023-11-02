import numpy as np
import pandas as pd
import ast
from typing import List


label2score = {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}


def get_real_label(label: str) -> int:
    """
    Convert label to real label based on mapping.

    Args:
        label (str): Label to be converted.

    Returns:
        int: Real label.
    """
    label = int(label)

    if label >= 3:
        return 1
    elif label == 2:
        return -1
    else:
        return 0


def preprocess_df(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Preprocess the DataFrame by renaming columns and updating labels.

    Args:
        df (pd.DataFrame): Input DataFrame.
        is_train (bool, optional): Whether the DataFrame is from a training set. Defaults to True.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    preprocessed_df = df.rename(
        columns={
            0: "topic_td",
            1: "topic_name",
            2: "sent_1",
            3: "sent_2",
            4: "label",
            5: "sent_1_tag",
            6: "sent_2_tag",
        }
    )

    if is_train:
        preprocessed_df.label = preprocessed_df.label.apply(
            lambda row: ast.literal_eval(row)[0]
        )
        preprocessed_df["is_paraphrase"] = preprocessed_df.apply(
            lambda row: get_real_label(row.label), axis=1
        )

    else:
        preprocessed_df["is_paraphrase"] = preprocessed_df.apply(
            lambda row: get_real_label(row.label), axis=1
        )
    return preprocessed_df


def get_data_as_list(df: pd.DataFrame) -> List:
    """
    Get data from the DataFrame as a list.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        List: List of data.
    """
    sentences_1 = df.sent_1.values.tolist()
    sentences_2 = df.sent_2.values.tolist()
    scores = df.label.apply(lambda row: label2score[row]).values.tolist()

    data_list = [
        (sentences_1[i], sentences_2[i], scores[i]) for i in range(len(sentences_1))
    ]
    return data_list
