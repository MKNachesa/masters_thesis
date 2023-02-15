# this script is a modification of Faton's script
import multiprocessing as mp
import sys
from pathlib import Path
import pandas as pd
import os
from functools import partial
os.chdir("../../riksdagen_anforanden")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from src.audio import split_audio_by_speech

df = pd.read_parquet("../masters_thesis/metadata/filtered_speeches.parquet")
df["filename"] = df["filename"].apply(lambda x: x.replace(".wav", ".mp3"))
df_groups = df.groupby("dokid")
df_groups = df_groups[["dokid", "anforande_nummer", "filename", "start_segment", "duration_segment"]]
df_list = [df_groups.get_group(x) for x in df_groups.groups]  # list of dfs, one for each dokid

pool = mp.Pool(24)

for dur in [60, 30, 10, 5, 3, 1]:
    partial_split_audio_by_speech = partial(split_audio_by_speech, segment_length=dur)

    df_dokids = pool.map(partial_split_audio_by_speech, tqdm(df_list, total=len(df_list)), chunksize=4)
    pool.close()

    df[f"filename_anforande_audio_{dur}"] = df[["start_segment", "duration_segment", "filename"]].apply(
        lambda x: Path(x["filename"]).parent
        / f"{Path(x['filename']).stem}_{x['start_segment']}_{x['start_segment'] + x['duration_segment']}.mp3",
        axis=1,
    )

    df[f"filename_anforande_audio_{dur}"] = df[f"filename_anforande_audio_{dur}"].apply(lambda x: str(x))
    df.to_parquet("../masters_thesis/metadata/filtered_speeches.parquet", index=False)
