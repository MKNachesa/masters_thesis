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

id_to_min_age = df.groupby("intressent_id").agg(
    lambda x: list(x)[0])[["min_age", "gender"]].to_dict(orient="index")

buckets = [set() for i in range(len(bucket_ranges))]

max_bucket_size = 4

bucket_genders = [{"F": 0, "M": 0} for i in range(len(bucket_ranges))]

# first pass for gender balance
for iid, d in id_to_min_age.items():
    min_age, gender = d.values()
    i = int((min_age-start)//5)
    if len(buckets[i]) < max_bucket_size \
       and bucket_genders[i][gender] < max_bucket_size//2:
        buckets[i].add(iid)
        bucket_genders[i][gender] += 1

# second pass to fill buckets (some buckets otherwise don't have 4 speakers)
for iid, d in id_to_min_age.items():
    min_age, gender = d.values()
    i = int((min_age-start)//5)
    if len(buckets[i]) < max_bucket_size and iid not in buckets[i]:
        buckets[i].add(iid)
        bucket_genders[i][gender] += 1

range_to_bucket = dict(zip(bucket_ranges, buckets))

ids_to_use = set(reduce(lambda x, y: x.union(y), range_to_bucket.values()))

df["to_use"] = df["intressent_id"].apply(lambda x: x in ids_to_use)

df_reduce = df[df.to_use==True].drop("to_use", axis=1)

df_reduce.to_parquet(df_bucketed, index=False)
