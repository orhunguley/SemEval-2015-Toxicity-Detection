import pandas as pd
import ast
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import evaluation
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import argparse
from utils import preprocess_df
from typing import List

label2score = {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}


def get_evaluator(test_df: pd.DataFrame) -> evaluation.EmbeddingSimilarityEvaluator:
    """
    Create an evaluator for sentence embeddings similarity.

    Args:
        test_df (pd.DataFrame): Test dataset.

    Returns:
        evaluation.EmbeddingSimilarityEvaluator: Evaluator for sentence similarity.
    """
    sentences_1 = test_df.sent_1.values.tolist()
    sentences_2 = test_df.sent_2.values.tolist()
    scores = test_df.label.apply(lambda row: label2score[row]).values.tolist()

    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        sentences_1,
        sentences_2,
        scores,
        write_csv=True,
        name="out_similarity_predictor",
    )

    return evaluator


def get_train_loader(train_df: pd.DataFrame) -> DataLoader:
    """
    Get a DataLoader for training data.

    Args:
        train_df (pd.DataFrame): Training dataset.

    Returns:
        DataLoader: DataLoader for training.
    """
    train_sentences_1 = train_df.sent_1.values.tolist()
    train_sentences_2 = train_df.sent_2.values.tolist()
    scores = train_df.label.apply(lambda row: label2score[row]).values.tolist()

    train_dataset = [
        InputExample(
            texts=[train_sentences_1[i], train_sentences_2[i]], label=scores[i]
        )
        for i in range(len(train_sentences_1))
    ]

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)

    return train_loader


def main(args: argparse.Namespace) -> None:
    """
    Main function for training a sentence similarity model.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    train_df = pd.read_csv(args.train_path, delimiter="\t", header=None)
    train_df = preprocess_df(train_df)

    test_df = pd.read_csv(args.test_path, delimiter="\t", header=None)
    test_df = preprocess_df(test_df, is_train=False)

    train_loader = get_train_loader(train_df=train_df)
    evaluator = get_evaluator(test_df=test_df)

    model = SentenceTransformer("all-mpnet-base-v2")

    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        evaluator=evaluator,
        evaluation_steps=args.evaluation_steps,
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
        default="./out_similarity_predictor/",
        help="Output path for the model",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=100, help="Number of warmup steps"
    )

    parser.add_argument(
        "--evaluation_steps", type=int, default=100, help="Number of evaluation steps"
    )
    args = parser.parse_args()
    
    main(args)
