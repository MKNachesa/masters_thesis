import pandas as pd
import os
from collections import defaultdict
from random import choice, sample
import pickle as pkl
from sklearn.metrics.pairwise import cosine_similarity
from functools import reduce
import matplotlib.pyplot as plt

NUM_PAIRS = 6

scripts_dir = os.getcwd()
os.chdir("../data")
data_dir = os.getcwd()
os.chdir("../results")
results_dir = os.getcwd()
os.chdir(scripts_dir)

speech_lengths = ["full", 60, 30, 10, 5, 3, 1]
# df = pd.read_parquet("../metadata/within_speaker_within_age_comparisons.parquet")
df = pd.read_parquet("../metadata/within_speaker_across_age_comparisons.parquet")
danfs_to_use = set()

for i in range(1, NUM_PAIRS+1):
    danfs_to_use.update(set(reduce(lambda x, y: list(x) + list(y), list(map(set, df[f"pair_{i}"].tolist())))))

dokids_to_use = set(i.split("_")[0] for i in danfs_to_use)

for speech_length in speech_lengths:
    print(f"Processing {speech_length}")
    length_dir = os.path.join(data_dir, str(speech_length))
    embs = dict()
#
    for emb_path in next(os.walk(length_dir))[2]:
        if emb_path.split("_")[1].split(".")[0] in dokids_to_use:
            with open(os.path.join(length_dir, emb_path), "rb") as infile:
                dokid_embs = pkl.load(infile)
            for danf, emb in dokid_embs.items():
                if danf in danfs_to_use:
                    embs[danf] = emb
#
    for i in range(1, NUM_PAIRS+1):
        df[f"score_{i}_{speech_length}"] = df[f"pair_{i}"].apply(lambda y: cosine_similarity(*list(map(lambda x: embs[x], y)))[0][0])
#
    df[f"score_mean_{speech_length}"] = df[[f"score_{i}_{speech_length}" for i in range(1, NUM_PAIRS+1)]].apply(lambda x: sum(x)/NUM_PAIRS, axis=1)

for speech_length in speech_lengths:
    plt.hist(df[f"score_mean_{speech_length}"].tolist())
    # plt.show()
    plt.savefig(os.path.join(results_dir, f"within_speaker_across_age_{speech_length}_cossim_score.png"))
    plt.close()



# across speakers
NUM_PAIRS = 1
speech_lengths = ["full", 60, 30, 10, 5, 3, 1]

df_across = pd.read_parquet("../metadata/across_speaker_comparisons_small.parquet")
# df_across["pair_1"] = df_across["pair1"]
# df_across.drop("pair1", inplace=True, axis=1)
danfs_to_use = set()
df_across[["danf1", "danf2"]] = pd.DataFrame(df_across["pair_1"].tolist(), index=df_across.index)

danfs_to_use = set(df_across.danf1.tolist()).union(set(df_across.danf2.tolist()))
dokids_to_use = set(map(lambda x: x.split("_")[0], set(df_across.danf1.tolist()))).union(set(map(lambda x: x.split("_")[0], set(df_across.danf2.tolist()))))

for speech_length in speech_lengths:
    print(f"Processing {speech_length}")
    length_dir = os.path.join(data_dir, str(speech_length))
    embs = dict()
#
    for emb_path in next(os.walk(length_dir))[2]:
        if emb_path.split("_")[1].split(".")[0] in dokids_to_use:
            with open(os.path.join(length_dir, emb_path), "rb") as infile:
                dokid_embs = pkl.load(infile)
            for danf, emb in dokid_embs.items():
                if danf in danfs_to_use:
                    embs[danf] = emb
#
    for i in range(1, NUM_PAIRS+1):
        df_across[f"score_{i}_{speech_length}"] = df_across[f"pair_{i}"].apply(lambda y: cosine_similarity(*list(map(lambda x: embs[x], y)))[0][0])
#
    # df_across[f"score_mean_{speech_length}"] = df_across[[f"score_{i}_{speech_length}" for i in range(1, NUM_PAIRS+1)]].apply(lambda x: sum(x)/NUM_PAIRS, axis=1)

for speech_length in speech_lengths:
    plt.hist(df_across[f"score_1_{speech_length}"].tolist())
    plt.savefig(os.path.join(results_dir, f"across_speaker_{speech_length}_cossim_score.png"))
    plt.close()

NUM_PAIRS = 6
for speech_length in speech_lengths:
    plt.hist(list(reduce(lambda x, y: x + y, [df[f"score_{i}_{speech_length}"].tolist() for i in range(1, NUM_PAIRS)])))
    plt.hist(df_across[f"score_1_{speech_length}"].tolist())
    plt.savefig(os.path.join(results_dir, f"within_speaker_across_age_VS_across_speaker_{speech_length}_cossim_score.png"))
    plt.close()