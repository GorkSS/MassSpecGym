import pandas as pd
import numpy as np
import torch

def _compress_spaces(s: str):
    new_s = ""
    for c in s:
        if c == " " and new_s[-1] != " ":
            new_s += c
        else:
            new_s += c

    return new_s

def _str2ndarray(s: str, sep=" ", dtype=np.float64):
    if isinstance(s, str):
        emb = _compress_spaces(s.replace("\n", "").replace("[", "").replace("]", "").strip())
        return np.fromstring(emb, sep=sep, dtype=dtype)
    else:
        return s

def _str2tensor(s: str, sep=",", dtype=np.float64):
    parts = s.replace("tensor([[", "").replace("]])", "").replace("\n", "").replace(" ", "").split(sep)
    return torch.from_numpy(_str2ndarray(" ".join(parts), dtype=dtype))

def read_tsv(path):
    return BambooDataFrame(pd.read_csv(path, sep="\t"))

class BambooDataFrame(pd.DataFrame):
    def col_to_ndarray(self, colname: str):
        self[colname] = self[colname].apply(lambda emb: _str2ndarray(emb))

    def col_to_tensor(self, colname: str):
        self[colname] = self[colname].apply(lambda emb: _str2tensor(emb))