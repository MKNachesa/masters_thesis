import pandas as pd
import os

script_dir = os.getcwd()
os.chdir("..")
thesis_dir = os.getcwd()
os.chdir("../riksdagen_anforanden/data/audio")
audio_dir = os.getcwd()
os.chdir(script_dir)

data_dir = os.path.join(thesis_dir, "data")
if "data" not in next(os.walk(thesis_dir))[1]:
    os.mkdir(data_dir)

metadata_dir = os.path.join(thesis_dir, "metadata")
bucket_file = os.path.join(metadata_dir, "bucketed_speeches.parquet")

df = pd.read_parquet(bucket_file)