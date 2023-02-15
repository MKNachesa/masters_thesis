# this script is a modification of Faton's script
import multiprocessing as mp
import sys
from pathlib import Path
import pandas as pd
import os
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
df_dokids = pool.map(split_audio_by_speech, tqdm(df_list, total=len(df_list)), chunksize=4)
pool.close()

df["filename_anforande_audio"] = df[["start_segment", "duration_segment", "filename"]].apply(
    lambda x: Path(x["filename"]).parent
    / f"{Path(x['filename']).stem}_{x['start_segment']}_{x['start_segment'] + x['duration_segment']}.mp3",
    axis=1,
)

df["filename_anforande_audio"] = df["filename_anforande_audio"].apply(lambda x: str(x))
df.to_parquet("../masters_thesis/metadata/filtered_speeches.parquet", index=False)
