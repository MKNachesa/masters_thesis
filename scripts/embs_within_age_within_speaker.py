import pandas as pd
import os
from collections import defaultdict
from random import choice, sample
import pickle as pkl
from sklearn.metrics.pairwise import cosine_similarity
from functools import reduce
from matplotlib.pyplot import pyplot

scripts_dir = os.getcwd()
os.chdir("../data")
data_dir = os.getcwd()
os.chdir(scripts_dir)
full_dir = os.path.join(data_dir, "full")

df = pd.read_parquet("../metadata/within_speaker_within_age_comparisons.parquet")

danfs_to_use = set()

for i in range(1, 7):
    danfs_to_use.update(set(reduce(lambda x, y: list(x) + list(y), list(map(set, df[f"pair_{i}"].tolist())))))

dokids_to_use = set(i.split("_")[0] for i in danfs_to_use)

embs = dict()

for emb_path in next(os.walk(full_dir))[2]:
    if emb_path.split("_")[1].split(".")[0] in dokids_to_use:
        with open(os.path.join(full_dir, emb_path), "rb") as infile:
            dokid_embs = pkl.load(infile)
        for danf, emb in dokid_embs.items():
            if danf in danfs_to_use:
                embs[danf] = emb

for i in range(1, 7):
    df[f"score_{i}"] = df[f"pair_{i}"].apply(lambda y: cosine_similarity(*list(map(lambda x: embs[x], y)))[0][0])

df["score_mean"] = df[[f"score_{i}" for i in range(1, 7)]].apply(lambda x: sum(x)/6, axis=1)

plt.hist(df.score_mean.tolist())
plt.show()