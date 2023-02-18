import numpy as np
import pandas as pd
from functools import partial

np.random.seed(1)

df = pd.read_parquet("../metadata/filtered_speeches.parquet")

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
  df[f"timestamps_{dur}"] = df[["start_segment", "end_segment"]].apply(
      partial_get_ts, axis=1
  )

df.to_parquet("../metadata/filtered_speeches_ts.parquet", index=False)
