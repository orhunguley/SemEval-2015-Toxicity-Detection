import numpy as np
import pandas as pd
import ast
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentencesDataset,
    InputExample,
    losses,
)
from sentence_transformers.cross_encoder.evaluation import (
    CEBinaryClassificationEvaluator,
)
from sentence_transformers.cross_encoder import CrossEncoder

from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)
from typing import List
from utils import get_data_as_list, preprocess_df
import argparse


def get_paraphrase_probs(
    model: CrossEncoder, sentence_pairs: List[List[str]], threshold: float = 0.5
) -> np.ndarray:
    """
    Get paraphrase probabilities using the provided model.

    Args:
        model (CrossEncoder): CrossEncoder model.
        sentence_pairs (List[List[str]]): List of sentence pairs.
        threshold (float, optional): Threshold for labeling. Defaults to 0.5.

    Returns:
        np.ndarray: Paraphrase probabilities.
    """
    preds = model.predict(sentence_pairs, convert_to_numpy=True)
    return preds


def get_similarities(
    model: SentenceTransformer, sentence_pairs: List[List[str]]
) -> np.ndarray:
    """
    Get cosine similarities using the provided model.

    Args:
        model (SentenceTransformer): SentenceTransformer model.
        sentence_pairs (List[List[str]]): List of sentence pairs.

    Returns:
        np.ndarray: Cosine similarities.
    """
    sentences_1 = [p[0] for p in sentence_pairs]
    sentences_2 = [p[1] for p in sentence_pairs]

    embeddings1 = model.encode(
        sentences_1,
        batch_size=len(sentences_1),
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    embeddings2 = model.encode(
        sentences_2,
        batch_size=len(sentences_1),
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))

    return cosine_scores


def get_labels(scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Get labels based on scores and threshold.

    Args:
        scores (np.ndarray): Scores to be thresholded.
        threshold (float, optional): Threshold for labeling. Defaults to 0.5.

    Returns:
        np.ndarray: Binary labels.
    """
    predicted_labels = np.copy(scores)
    predicted_labels[predicted_labels >= 0.5] = True
    predicted_labels[predicted_labels < 0.5] = False

    return predicted_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="User-Generated Content Moderation")
    parser.add_argument(
        "--output-file",
        type=str,
        default="systemoutputs/PIT2015_transformer_scores.output",
        help="Output file path",
    )

    args = parser.parse_args()

    paraphrase_model = CrossEncoder("out_paraphrase_detector/")

    test_df = pd.read_csv("data/test.data", delimiter="\t", header=None)
    test_df = preprocess_df(test_df, is_train=False)

    data_list = get_data_as_list(test_df)
    sentence_pairs = [[pair[0], pair[1]] for pair in data_list]

    preds = get_paraphrase_probs(paraphrase_model, sentence_pairs)
    pred_labels = get_labels(scores=preds, threshold=0.5)

    similarity_model = SentenceTransformer("out_similarity_predictor/")
    similarity_scores = get_similarities(similarity_model, sentence_pairs)

    pred_labels = pred_labels.astype(bool).astype(str)
    pred_df = pd.DataFrame(
        {"is_paraphrase": pred_labels, "similarity_score": similarity_scores}
    )
    pred_df.is_paraphrase = pred_df.is_paraphrase.apply(lambda x: str(x).lower())
    pred_df.to_csv(
        args.output_file,
        header=None,
        index=None,
        sep="\t",
        mode="w",
    )
