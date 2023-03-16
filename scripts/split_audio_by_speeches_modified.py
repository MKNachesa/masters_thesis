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
import datetime
import time
import datetime
# import nemo.collections.asr as nemo_asr

##--------------------------------------------#
# first run cant_open_debate.py!
# use the output to ignore large files, process those manually
##--------------------------------------------#

##speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
##    model_name='titanet_large')

def to_timestamp(millitime):
    return str(datetime.timedelta(seconds=datetime.timedelta(milliseconds=millitime).seconds))


if __name__ == "__main__":
    scripts_dir = os.getcwd()
    os.chdir("..")
    thesis_dir = os.getcwd()
    os.chdir("scripts")
    vp_dir = os.path.join(thesis_dir, "data")
    # speeches_file = os.path.join(thesis_dir, "metadata/bucketed_speeches.parquet")
    speeches_file = os.path.join(thesis_dir, "metadata/all_speeches_ts_downsize.parquet")
    # peeches_file = os.path.join(thesis_dir, "metadata/filtered_speeches_ts.parquet")
    # processed_debates_file = os.path.join(thesis_dir, "metadata/processed_debates.pkl")

    riksdag_dir = "/data/datasets/riksdagen_anforanden"
    audio_dir = os.path.join(riksdag_dir, "data/audio")

    os.chdir(scripts_dir)

    # sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent)+"/riksdagen_anforanden")

    from tqdm import tqdm

    from src.audio import split_audio_by_speech

    df = pd.read_parquet(speeches_file)
    df = df.drop_duplicates("dokid_anfnummer").reset_index(drop=True)
    # processed_debates = pkl.load(
    #     open(processed_debates_file, "rb"))
    # all_dokids = set(df.dokid.tolist())
    # dokid_to_process = list(all_dokids.difference(processed_debates))

    # REMOVE ME TO RUN ON EVERYTHING
    #-------------------------------
    # dokid_to_ignore = ['GZ01FiU1', 'GT01UbU1']  # big files, python can handle them
    # df = df[df.dokid.apply(lambda x: x not in dokid_to_ignore)]
    # df = df[df.dokid.apply(lambda x: x in dokid_to_process)].reset_index()
    # df = df[:19]

    # df = df[(df.dokid=="GZ01FiU1") | (df.dokid=="GT01UbU1")]
    # df = df[df.dokid=="GT01UbU1"]
    #-------------------------------

    df["filename"] = df["filename"].apply(lambda x: x.replace(".wav", ".mp3"))

    durations = [1, 3, 5, 10, 30, 60, "full"]

    for dur in durations:
        df[f"split_timestamps_{dur}"] = df[f"timestamps_{dur}"].apply(lambda x: [to_timestamp(x[0]), to_timestamp(x[1])])

    # for dur in [None]:#, 60, 30, 10, 5, 3, 1]:#
    # for dur in [10]:    #, 5, 3, 1]:# None, 60, 30, 
    # dur = 5
    df_groups = df.groupby("dokid")
    # dur_str = dur if dur else "full"
    df_groups = df_groups[["dokid", "anforande_nummer", "filename", "dokid_anfnummer"]
                          + [f"timestamps_{dur}" for dur in durations]
                          + [f"split_timestamps_{dur}" for dur in durations]]
    df_list = [df_groups.get_group(x) for x in df_groups.groups]  # list of dfs, one for each dokid
    pool = mp.Pool(24)

    start = time.time()

    partial_split_audio_by_speech = partial(split_audio_by_speech,
##                                                speaker_model=speaker_model,
                                            vp_dir=vp_dir,
                                            # segment_length=dur,
                                            audio_dir=audio_dir)

    df_dokids = pd.concat(pool.map(partial_split_audio_by_speech, tqdm(df_list, total=len(df_list)), chunksize=4), axis=0)
    pool.close()

    end = time.time()

    print(f"Processing {len(df)} speeches took {to_timestamp((end-start)*1000)}")

    # df[f"filename_anforande_audio_{dur_str}"] = df_dokids[f"filename_anforande_audio_{dur_str}"]

    # print(df[[f"filename_anforande_audio_{dur_str}", "dokid", "shortname"]])

        # not really needed as timestamps already saved
        # df.to_parquet(speeches_file, index=False)
