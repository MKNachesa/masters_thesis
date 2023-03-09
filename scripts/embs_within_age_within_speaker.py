import pandas as pd
import os
from collections import defaultdict
from random import choice, sample
import pickle as pkl
from sklearn.metrics.pairwise import cosine_similarity
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np

scripts_dir = os.getcwd()
os.chdir("../data")
data_dir = os.getcwd()
os.chdir("../results")
results_dir = os.getcwd()
os.chdir("../metadata")
meta_dir = os.getcwd()
os.chdir(scripts_dir)

def get_cossim_df(num_pairs, parquet_path, save_path):
    print(f"Processing {save_path}")

    NUM_PAIRS = num_pairs
    df = pd.read_parquet(os.path.join(meta_dir, parquet_path))
#
    danfs_to_use = set()
    for i in range(1, NUM_PAIRS+1):
        danfs_to_use.update(set(reduce(lambda x, y: list(x) + list(y), list(map(set, df[f"pair_{i}"].tolist())))))
#
    dokids_to_use = set(i.split("_")[0] for i in danfs_to_use)
#
    for speech_length in speech_lengths:
        print(f"Processing {speech_length}")
        length_dir = os.path.join(data_dir, str(speech_length))
        embs = dict()
        for emb_path in next(os.walk(length_dir))[2]:
            if emb_path.split("_")[1].split(".")[0] in dokids_to_use:
                with open(os.path.join(length_dir, emb_path), "rb") as infile:
                    dokid_embs = pkl.load(infile)
                for danf, emb in dokid_embs.items():
                    if danf in danfs_to_use:
                        embs[danf] = emb
        for i in range(1, NUM_PAIRS+1):
            df[f"score_{i}_{speech_length}"] = df[f"pair_{i}"].apply(lambda y: cosine_similarity(*list(map(lambda x: embs[x], y)))[0][0])
        df[f"score_mean_{speech_length}"] = df[[f"score_{i}_{speech_length}" for i in range(1, NUM_PAIRS+1)]].apply(lambda x: sum(x)/NUM_PAIRS, axis=1)
#
    for speech_length in speech_lengths:
        plt.hist(df[f"score_mean_{speech_length}"].tolist())
        plt.title(f'{(" ").join(save_path.split("_"))} cosine similarity scores for {speech_length} speech length')
        plt.savefig(os.path.join(results_dir, f"{save_path}_{speech_length}_cossim_score.png"))
        plt.close()
#
    lesbian_flag = [(213, 45, 0), (239, 118, 39), (255, 154, 86), (230, 230, 230), (209, 98, 164), (181, 86, 144), (163, 2, 98)]
    for i, speech_length in enumerate(speech_lengths):
        plt.hist(df[f"score_mean_{speech_length}"].tolist(), label=f"{speech_length}",
        bins=np.arange(0, 1, 0.02), color=tuple(map(lambda x: x/255, lesbian_flag[i]))),
    plt.title(f'{(" ").join(save_path.split("_"))} cosine similarity scores')
    plt.legend()
    plt.savefig(os.path.join(results_dir, f"{save_path}_all_cossim_score.png"))
    plt.close()
    
    print()
    return df

speech_lengths = ["full", 60, 30, 10, 5, 3, 1]

across_age_df = get_cossim_df(num_pairs=3,
                              parquet_path="within_speaker_across_age_comparisons.parquet",
                              save_path="within_speaker_across_age")

within_age_df = get_cossim_df(num_pairs=3,
                              parquet_path="within_speaker_within_age_comparisons.parquet",
                              save_path="within_speaker_within_age")

across_speaker_df = get_cossim_df(num_pairs=1,
                                  parquet_path="across_speaker_comparisons.parquet",
                                  save_path="across_speaker")

NUM_PAIRS = 3
for speech_length in speech_lengths:
    plt.hist(list(reduce(lambda x, y: x + y, [across_age_df[f"score_{i}_{speech_length}"].tolist() for i in range(1, NUM_PAIRS)])),
             label="within speakers across ages")
    plt.hist(across_speaker_df[f"score_1_{speech_length}"].tolist(),
             label="across speakers")
    plt.legend()
    plt.title(f'across speaker VS within speaker across ages cosine similarity scores for {speech_length} speech length')
    plt.savefig(os.path.join(results_dir, f"withiin_speaker_across_age_VS_across_speaker_{speech_length}_cossim_score.png"))
    plt.close()

for speech_length in speech_lengths:
    plt.hist(list(reduce(lambda x, y: x + y, [within_age_df[f"score_{i}_{speech_length}"].tolist() for i in range(1, NUM_PAIRS)])),
             label="within speakers within ages")
    plt.hist(across_speaker_df[f"score_1_{speech_length}"].tolist(),
             label="across speakers")
    plt.legend()
    plt.title(f'across speaker VS within speaker within ages cosine similarity scores for {speech_length} speech length')
    plt.savefig(os.path.join(results_dir, f"within_speaker_within_age_VS_across_speaker_{speech_length}_cossim_score.png"))
    plt.close()