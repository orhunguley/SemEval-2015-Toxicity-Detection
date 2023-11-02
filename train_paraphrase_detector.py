import numpy as np
import pandas as pd
import ast

import torch
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
)
from sentence_transformers.cross_encoder import CrossEncoder
from paraphrase_evaluator import ParaphraseEvaluator
from utils import preprocess_df
from typing import List
import argparse

label2score = {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}


def get_paraphrase_evaluator(test_df: pd.DataFrame) -> ParaphraseEvaluator:
    """
    Create a ParaphraseEvaluator for paraphrase detection.

    Args:
        test_df (pd.DataFrame): Test dataset.

    Returns:
        ParaphraseEvaluator: Evaluator for paraphrase detection.
    """
    sentences_1 = test_df.sent_1.values.tolist()
    sentences_2 = test_df.sent_2.values.tolist()
    labels = test_df.is_paraphrase.values.tolist()

    s_pairs = [[sentences_1[i], sentences_2[i]] for i in range(len(sentences_1))]
    evaluator = ParaphraseEvaluator(
        s_pairs,
        labels,
        write_csv=True,
        name="paraphrase",
    )

    return evaluator


def get_paraphrase_train_loader(train_df: pd.DataFrame) -> DataLoader:
    """
    Get a DataLoader for paraphrase detection training.

    Args:
        train_df (pd.DataFrame): Training dataset.

    Returns:
        DataLoader: DataLoader for training.
    """
    train_sentences_1 = train_df.sent_1.values.tolist()
    train_sentences_2 = train_df.sent_2.values.tolist()
    labels = train_df.is_paraphrase.values.tolist()

    train_dataset = [
        InputExample(
            texts=[train_sentences_1[i], train_sentences_2[i]], label=labels[i]
        )
        for i in range(len(train_sentences_1))
    ]

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)

    return train_loader


def main(args):
    torch.multiprocessing.set_start_method("spawn")

    train_df = pd.read_csv(args.train_path, delimiter="\t", header=None)
    train_df = preprocess_df(train_df)
    train_df = train_df[train_df.is_paraphrase.isin([0, 1])]

    test_df = pd.read_csv(args.test_path, delimiter="\t", header=None)
    test_df = preprocess_df(test_df, is_train=False)
    test_df = test_df[test_df.is_paraphrase.isin([0, 1])]

    train_loader = get_paraphrase_train_loader(train_df=train_df)
    evaluator = get_paraphrase_evaluator(test_df=test_df)

    model = CrossEncoder("distilroberta-base", num_labels=1)

    model.fit(
        train_dataloader=train_loader,
        epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        evaluator=evaluator,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/train.data",
        help="Path to training data",
    )
    parser.add_argument(
        "--test_path", type=str, default="data/test.data", help="Path to test data"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./out_paraphrase_detector_test/",
        help="Output path for the model",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=100, help="Number of warm-up steps"
    )

    args = parser.parse_args()
    main(args)
