# this script is a modification of Faton's script
import multiprocessing as mp
# import multiprocess as mp
# from multiprocessing import get_context
import sys
from pathlib import Path
import pandas as pd
import os
from functools import partial
import pickle as pkl
import numpy as np
import nemo.collections.asr as nemo_asr

##--------------------------------------------#
#| Make sure to import and download all       |
#| necessary modules!                         |
#|                                            |
#| Check all comments first!                  |
#|                                            |
#| This script only runs on a subset of files |
##--------------------------------------------#

##speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
##    model_name='titanet_large')

if __name__ == "__main__":
    scripts_dir = os.getcwd()
    os.chdir("..")
    thesis_dir = os.getcwd()
    os.chdir("scripts")
    vp_dir = os.path.join(thesis_dir, "data")
    speeches_file = os.path.join(thesis_dir, "metadata/filtered_speeches_ts.parquet")
    processed_debates_file = os.path.join(thesis_dir, "metadata/processed_debates.pkl")

    os.chdir("../../riksdagen_anforanden")

    riksdag_dir = os.getcwd()
    audio_dir = os.path.join(riksdag_dir, "data/audio")

    os.chdir(scripts_dir)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from tqdm import tqdm

    from src.audio import split_audio_by_speech

    df = pd.read_parquet(speeches_file)
    # processed_debates = pkl.load(
    #     open(processed_debates_file, "rb"))
    # all_dokids = set(df.dokid.tolist())
    # dokid_to_process = list(all_dokids.difference(processed_debates))

    # REMOVE ME TO RUN ON EVERYTHING
    #-------------------------------
    dokid_to_process = ['GR01LU22', 'GR01UU12', 'GR10476', 'GR10508', 'GR10529', 
                        'GR10530', 'GR10523', 'GR10533', 'GR01TU10', 'GR10197']
    df = df[:19]
    #-------------------------------
    df["filename"] = df["filename"].apply(lambda x: x.replace(".wav", ".mp3"))

    for dur in [None, 60, 30, 10, 5, 3, 1]:#
        df_groups = df.groupby("dokid")
        dur_str = dur if dur else "full"
        df_groups = df_groups[["dokid", "anforande_nummer", "filename", f"timestamps_{dur_str}", "dokid_anfnummer"]]
        df_list = [df_groups.get_group(x) for x in df_groups.groups]  # list of dfs, one for each dokid
        pool = mp.Pool(24)

        partial_split_audio_by_speech = partial(split_audio_by_speech,
##                                                speaker_model=speaker_model,
                                                vp_dir=vp_dir,
                                                segment_length=dur,
                                                audio_dir=audio_dir)

        df_dokids = pd.concat(pool.map(partial_split_audio_by_speech, tqdm(df_list, total=len(df_list)), chunksize=4), axis=0)
        pool.close()

        df[f"filename_anforande_audio_{dur_str}"] = df_dokids[f"filename_anforande_audio_{dur_str}"]

        print(df[[f"filename_anforande_audio_{dur_str}", "dokid", "shortname"]])

        # not really needed as timestamps already saved
        # df.to_parquet(speeches_file, index=False)
