import os
from pathlib import Path
from pydub import AudioSegment
from nltk import sent_tokenize
import numpy as np


def split_audio_by_speech(df, start_key="start_segment", duration_key="duration_segment",
                          audio_dir="data/audio", file_exists_check=False, segment_length=None):
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

    filename_dokid = df["filename"].iloc[0]
    print(filename_dokid)
    segments = df[[start_key, duration_key]].to_dict(orient="records")
    sound = AudioSegment.from_mp3(os.path.join(audio_dir, filename_dokid))
    sound = sound.set_frame_rate(16000)
    sound = sound.set_channels(1)

    filenames_speeches = []
    for segment in segments:
        # cut off 10 secs from start and end
        start = (float(segment[start_key]) + 10) * 1000  # ms
        end = (float(segment[start_key]) + float(segment[duration_key]) - 10) * 1000

        if segment_length:
            start = np.random.uniform(start, end - (segment_length*1000))
            end = start + segment_length * 1000
        split = sound[start:end]

        filename = (
            Path(filename_dokid).parent / Path(filename_dokid).stem
        )  # Filename without extension.

        filename_speech = Path(
            f"{filename}_{start}_{end}.wav"
        )

        if file_exists_check:
            if os.path.exists(os.path.join(audio_dir, filename_speech)):
                print(f"File {filename_speech} already exists.")
                continue

        filenames_speeches.append(filename_speech)
        split.export(os.path.join(audio_dir, filename_speech), format="wav")

    segment_length = segment_length if segment_length else "full"
    df[f"filename_anforande_audio_{segment_length}"] = filenames_speeches
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
