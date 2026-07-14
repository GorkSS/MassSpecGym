from pandas import DataFrame
from typing import List
import numpy as np

def cosine_similarity(df : DataFrame, reference_vec_column : str, pred_vec_column : str) -> List:
    return [np.dot(df[reference_vec_column].iloc[i], df[pred_vec_column].iloc[i].flatten()) / (np.linalg.norm(df[reference_vec_column].iloc[i]) * np.linalg.norm(df[pred_vec_column].iloc[i].flatten())) for i in range(len(df))]

def euclidean_distance(df : DataFrame, reference_vec_column : str, pred_vec_column : str) -> List:
    return [np.linalg.norm(df[reference_vec_column].iloc[i] - df[pred_vec_column].iloc[i].flatten().numpy()) for i in range(len(df))]