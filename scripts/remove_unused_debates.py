import os
import shutil
import pandas as pd

filtered_file = "../metadata/filtered_speeches_ts.parquet"
df = pd.read_parquet(filtered_file)
print(df)
dokids = set(df.dokid.tolist())

os.chdir("../../riksdagen_anforanden/data/audio")

for dir in next(os.walk("."))[1]:
  if dir not in dokids:
    shutil.rmtree(dir)