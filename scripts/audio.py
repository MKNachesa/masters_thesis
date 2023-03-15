import os
from pathlib import Path
from pydub import AudioSegment
from nltk import sent_tokenize
import pickle as pkl
import numpy as np
import subprocess
# import nemo.collections.asr as nemo_asr

def split_audio_by_speech(df, vp_dir,#speaker_model=None,
                          audio_dir="data/audio", 
                          file_exists_check=False, segment_length=None):
    """
    Split audio file by anförande (speech) and save to disk in folder for specific dokid.

    Parameters:
        df (pandas.DataFrame): Subset of DataFrame with audio metadata for specific dokid.
            df["filename"] looks like "H901KrU5/2442204200009516121_aud.mp3",
            i.e. {dokid}/{filename}.
        audio_dir (str): Path to directory where audio files should be saved.
        file_exists_check (bool): If True, checks whether split file already exists and
            skips it. When False, reprocesses all files.
    """

    # speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
    #     model_name='titanet_large')
    
    filename_dokid = df["filename"].iloc[0]
    filename = (
        Path(filename_dokid).parent / Path(filename_dokid).stem
    )  # Filename without extension.
    input_file = os.path.join(audio_dir, filename_dokid)

    try:
        sound = AudioSegment.from_mp3(os.path.join(audio_dir, filename_dokid))
        sound = sound.set_frame_rate(16000)
        sound = sound.set_channels(1)
    except:
        return df

    for segment_length in ["full", 60, 30, 10, 5, 3, 1]:
        # segment_length = segment_length if segment_length else "full"
        segments = df[[f"timestamps_{segment_length}", f"split_timestamps_{segment_length}"]].to_dict(orient="records")

        filenames_speeches = []

        for segment in segments:
            start = float(segment[f"timestamps_{segment_length}"][0])  # ms
            end = float(segment[f"timestamps_{segment_length}"][1])

            timestamp_start, timestamp_end = segment[f"split_timestamps_{segment_length}"]

            filename_speech = Path(
                f"{filename}_{start}_{end}.wav"
            )

            filenames_speeches.append(filename_speech)

            output_file = os.path.join(audio_dir, filename_speech)

            if file_exists_check:
                if os.path.exists(os.path.join(audio_dir, filename_speech)):
                    print(f"File {filename_speech} already exists.")
                    continue

            split = sound[start:end]
            split.export(os.path.join(audio_dir, filename_speech), format="wav")
            # retcode = subprocess.call(["ffmpeg", "-i", input_file, "-ac", "1", "-ss", timestamp_start, "-to", timestamp_end, "-c", "copy", output_file],
            #           stdout=subprocess.DEVNULL,
            #           stderr=subprocess.STDOUT
            # )

        df[f"filename_anforande_audio_{segment_length}"] = filenames_speeches
    # dok_to_emb = dict()
    # dokid = df.dokid.iloc[0]

    # for i, row in df.iterrows():
    #     f = row[f"filename_anforande_audio_{segment_length}"]
    #     dok = row.dokid_anfnummer
    #     file_path = os.path.join(audio_dir, f)
    #     emb = speaker_model.get_embedding(file_path)
    #     dok_to_emb[dok] = emb
    #     os.remove(file_path)
    # dur_dir = os.path.join(vp_dir, f"{segment_length}")
    # if f"{segment_length}" not in next(os.walk(vp_dir))[1]:
    #     os.mkdir(dur_dir)
    # f = open(os.path.join(dur_dir, f"emb_{dokid}.pkl"), "wb")
    # pkl.dump(dok_to_emb, f)
    # f.close()

    print(f"{filename_speech.parent} complete", end="\r", flush=True)
    return df


def get_corrupt_audio_files(df, audio_dir="data/audio", return_subset=True):
    """
    Get list of corrupt audio files that were not able to be force aligned.
    We retry those mp3 files that have no corresponding json sync file.

    Parameters:
        df (pandas.DataFrame): DataFrame with all audio metadata, including
        filenames of audio files.
        audio_dir (str): Path to directory where audio files were saved.
        return_subset (bool): If True, returns subset of df with corrupt audio files.
            If False, returns entire df with column "corrupt" indicating whether
            audio file is corrupt or not.
    """

    def json_exists(filename):
        return os.path.exists(
            Path(audio_dir) / Path(filename).parent / f"{Path(filename).stem}.json"
        )

    df["corrupt"] = df["filename_anforande_audio"].apply(lambda x: not json_exists(x))

    if return_subset:
        return df[df["corrupt"]].reset_index(drop=True)
    else:
        return df


def split_text_by_speech(df, text_dir="data/audio"):
    """
    Split text file by anforande (speech) and save to disk in folder for specific dokid.
    If audio file for speech is saved as "H901KrU5/2442204200009516121_aud_0_233.mp3",
    then text file should be saved as "H901KrU5/2442204200009516121_aud_0_233.txt".

    Assumes split_audio_by_speech() has been run.

    Parameters:
        df (pandas.DataFrame): Subset of DataFrame with audio metadata for specific dokid.
            df["anftext"] contains text for each speech.
        text_dir (str): Path to directory where text files should be saved. Default to same
            directory as audio files.
    """

    df["lines"] = df["anftext"].apply(lambda x: sent_tokenize(x) if x is not None else [None])
    df["filename_anforande_text"] = df["filename_anforande_audio"].apply(
        lambda x: Path(x).parent / f"{Path(x).stem}.txt"
    )

    for _, row in df.iterrows():

        if row["lines"][0] is None:
            continue

        with open(os.path.join(text_dir, row["filename_anforande_text"]), "w") as f:
            for line in row["lines"]:
                f.write(line + "\n")

    return df
