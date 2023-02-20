import pandas as pd
import os
from functools import reduce

os.chdir("..")
metadata = os.path.join(os.getcwd(), "metadata")
df_ts = os.path.join(metadata, "filtered_speeches_ts.parquet")
df_bucketed = os.path.join(metadata, "bucketed_speeches.parquet")
os.chdir("scripts")

df = pd.read_parquet("../metadata/filtered_speeches_ts.parquet")

df["min_age"] = df.groupby(["intressent_id"])["age"].transform("min")

df_age_group_count = df.groupby("intressent_id").agg("mean").groupby("min_age").agg("count")

start = df["min_age"].min() # 24
end = df["min_age"].max()   # 73
bucket_ranges = [(i, i+4) for i in range(start, end, 5)]

id_to_min_age = df.groupby("intressent_id").agg("mean")["min_age"].to_dict()

buckets = [list() for i in range(len(bucket_ranges))]

for iid, min_age in id_to_min_age.items():
    i = int((min_age-start)//5)
    if len(buckets[i]) < 4:
        buckets[i].append(iid)

range_to_bucket = dict(zip(bucket_ranges, buckets))

ids_to_use = set(reduce(lambda x, y: x + y, range_to_bucket.values()))

df["to_use"] = df["intressent_id"].apply(lambda x: x in ids_to_use)

df_reduce = df[df.to_use==True].drop("to_use", axis=1)

df_reduce.to_parquet(df_bucketed)
