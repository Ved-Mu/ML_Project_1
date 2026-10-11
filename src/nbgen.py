"""Tiny helper to write .ipynb files with the standard library only."""
import json


class NB:
    def __init__(self):
        self.cells = []

    def md(self, text):
        self.cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})

    def code(self, text):
        self.cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": text.strip("\n")})

    def save(self, path):
        for i, c in enumerate(self.cells):
            c["id"] = f"cell-{i:02d}"
        nb = {
            "cells": self.cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("wrote", path, "with", len(self.cells), "cells")


LOAD_CODE = '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 30)

DATA_DIR = "../data/raw/CMAPSSData"   # on Colab: folder where you uploaded the 3 files

index_cols   = ["unit", "cycle"]
setting_cols = ["setting_1", "setting_2", "setting_3"]
sensor_cols  = [f"s{i}" for i in range(1, 22)]
col_names    = index_cols + setting_cols + sensor_cols

train = pd.read_csv(f"{DATA_DIR}/train_FD001.txt", sep=r"\\s+", header=None, names=col_names)
test  = pd.read_csv(f"{DATA_DIR}/test_FD001.txt",  sep=r"\\s+", header=None, names=col_names)
rul   = pd.read_csv(f"{DATA_DIR}/RUL_FD001.txt",   sep=r"\\s+", header=None, names=["true_RUL"])
print("train:", train.shape, "| test:", test.shape)
'''
