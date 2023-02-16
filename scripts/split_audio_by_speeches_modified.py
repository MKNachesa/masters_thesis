# this script is a modification of Faton's script
import multiprocessing as mp
import sys
from pathlib import Path
import pandas as pd
import os
from functools import partial
import pickle as pkl
os.chdir("../../riksdagen_anforanden")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from src.audio import split_audio_by_speech

df = pd.read_parquet("../masters_thesis/metadata/filtered_speeches.parquet")
processed_debates = pkl.load("../masters_thesis/metadata/processed_debates")
all_dokids = set(df.dokid.tolist())
dokid_to_process = list(all_dokids.difference(processed_debates))
# df = df[:10]
df["filename"] = df["filename"].apply(lambda x: x.replace(".wav", ".mp3"))
df = df[(df[~df.isna()]) & (df[df.valid_audio==True])]

df_groups = df.groupby("dokid")
df_groups = df_groups[["dokid", "anforande_nummer", "filename", "start_segment", "duration_segment"]]
df_list = [df_groups.get_group(x) for x in df_groups.groups]  # list of dfs, one for each dokid

for dur in [None]:#, 60, 30, 10, 5, 3, 1]:
    pool = mp.Pool(24)

    partial_split_audio_by_speech = partial(split_audio_by_speech, segment_length=dur)

    df_dokids = pd.concat(pool.map(partial_split_audio_by_speech, tqdm(df_list, total=len(df_list)), chunksize=4), axis=0)
    dur = dur if dur else "full"
    pool.close()

    # df[f"filename_anforande_audio_{dur}"] = df[["start_segment", "duration_segment", "filename"]].apply(
    #     lambda x: Path(x["filename"]).parent
    #     / f"{Path(x['filename']).stem}_{x['start_segment']}_{x['start_segment'] + x['duration_segment']}.wav",
    #     axis=1,
    # )

    # df[f"filename_anforande_audio_{dur}"] = df[f"filename_anforande_audio_{dur}"].apply(lambda x: str(x))
    df[f"filename_anforande_audio_{dur}"] = df_dokids[f"filename_anforande_audio_{dur}"]

    print(df[[f"filename_anforande_audio_{dur}", "dokid", "shortname"]])
    df.to_parquet("../masters_thesis/metadata/filtered_speeches.parquet", index=False)

for dokid in dokid_to_process:

