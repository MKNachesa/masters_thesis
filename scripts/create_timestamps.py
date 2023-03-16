import numpy as np
import pandas as pd
from functools import partial

np.random.seed(1)

df_full = pd.read_parquet("../metadata/riksdagen_speeches_with_ages.parquet")
df_ts = pd.read_parquet("../metadata/filtered_speeches_ts.parquet")
df_full = df_full[~df_full.start_segment.isna()].reset_index(drop=True)
df_full = df_full.drop_duplicates(["dokid_anfnummer"]).reset_index(drop=True)
df_ts = df_ts.drop_duplicates(["dokid_anfnummer"]).reset_index(drop=True)
df_full = df_full[df_full.duration_segment >= 80].reset_index(drop=True)


def get_ts(start_end, dur):
  start, end = start_end
  if not dur:
    return ((start+10)*1000, (end-10)*1000)
  else:
    pair = [np.random.uniform(start+10, end-dur-10)*1000]*2
    pair[1] += dur*1000
    return tuple(pair)

for dur in [None, 60, 30, 10, 5, 3, 1]:
  partial_get_ts = partial(get_ts, dur=dur)
  if not dur:
    dur = "full"
  df_full[f"timestamps_{dur}"] = df_full[["start_segment", "end_segment"]].apply(
      partial_get_ts, axis=1
  )

df_full = df_full.set_index("dokid_anfnummer")
df_ts = df_ts.set_index("dokid_anfnummer")
df_full.update(df_ts)
df_full.reset_index(inplace=True)

df_full.to_parquet("../metadata/all_speeches_ts.parquet", index=False)
cols_to_keep = ["dokid_anfnummer", "dokid", "anforande_nummer", "filename"]
for dur in ["full", 60, 30, 10, 5, 3, 1]:
  cols_to_keep.append(f"timestamps_{dur}")

df_downsize = df_full[cols_to_keep].reset_index(drop=True)
df_downsize.to_parquet("../metadata/all_speeches_ts_downsize.parquet", index=False)