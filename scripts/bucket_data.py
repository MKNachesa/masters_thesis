import pandas as pd
import os
from functools import reduce
from collections import defaultdict


def quality_filter(df):
    # Speeches quality filtering
    #
    # get rid of speakers with no shortname, birthyear,
    # and who spoke less than 10 years
    # automatically gets rid of lines with no intressent_id
    df_filt = df[(~df["shortname"].isna()) & (df["birthyear"] != 0)]
    #
    # deduplicate rows
    df_filt = df_filt.drop_duplicates(
        ["dokid_anfnummer", "intressent_id", "start_segment"])
    #
    # get rid of invalid audio files and duplicate speeches
    df_filt = df_filt[(df_filt["count_dokid_anfnummer"] == 1)
                    & (df_filt["valid_audio"] == True)
                    & (~df_filt["start_segment"].isna())]
    #
    # only keep speeches with 1 speech segment
    df_filt = df_filt[df_filt["nr_speech_segments"] == 1.0]
    #
    # keep speeches within this length ratio
    df_filt = df_filt[(df_filt["length_ratio"] > 0.7)
                    & (df_filt["length_ratio"] < 1.3)]
    #
    # keep speeches within this overlap ratio
    df_filt = df_filt[(df_filt["overlap_ratio"] > 0.7)
                    & (df_filt["overlap_ratio"] < 1.3)]
    #
    # exclude speeches where speakers mention themselves (probably wrong name)
    df_filt = df_filt[df_filt[["anftext", "shortname"]].apply(
        lambda x: x.shortname not in x.anftext.lower(), axis=1)]
    #
    # reset index
    df_filt = df_filt.reset_index(drop=True)
    #
    # get rid of non-monotonically increasing speeches
    print("Getting rid of non-monotonically following speeches")
    dokids = set(df_filt["dokid"].tolist())
    non_monotonic = set()
    for i, dokid in enumerate(dokids):
        if (i+1) % 1000 == 0:
            print(f"Processed {i+1}/{len(dokids)} debates", flush=True)
        mini_df = df_filt[df_filt["dokid"] == dokid]
        for i, row in mini_df[1:].iterrows():
            anf = row["anforande_nummer"]
            prev_anf = mini_df.loc[i-1]["anforande_nummer"]
            if prev_anf > anf:
                anf = row["dokid_anfnummer"]
                prev_anf = mini_df.loc[i-1]["dokid_anfnummer"]
                non_monotonic.add(prev_anf)
                non_monotonic.add(anf)
                #
    df_filt = df_filt[df_filt["dokid_anfnummer"].apply(lambda x: x not in non_monotonic)]
    #
    # only keep speakers who have at least 3 speeches every year
    at_least_3 = df_filt.groupby(
        ["intressent_id", "age"]).agg("count")["party"].to_dict()
    df_filt["at_least_3"] = df_filt[["intressent_id", "age"]].apply(
        lambda x: at_least_3[(x.intressent_id, x.age)] >= 3, axis=1)
    df_filt = df_filt[df_filt["at_least_3"] == True]
    #
    # reset index
    df = df_filt.reset_index(drop=True)
    return df


def select_ids(df, max_bucket_size):
    df["min_age"] = df.groupby(["intressent_id"])["age"].transform("min")
    #
    start = df["min_age"].min() # 24
    end = df["min_age"].max()   # 67
    bucket_ranges = [(i, i+4) for i in range(start, end, 5)]
    #
    # counting number of speakers per gender and age bucket -------------------------------------------------------
    df["min_age_bucket"] = df.min_age.apply(lambda x: f"{(x-start)//5*5+start}-{(x-start)//5*5+start+4}") 
    df["min_age_buck_num"] = df.min_age.apply(lambda x: (x-start)//5*5+start) 
    df["gender_num"] = df.gender.apply(lambda x: 0 if x=="F" else 1)
    #
    df_bucket_gender_count = df.groupby("intressent_id").agg("mean").groupby(["min_age_buck_num", "gender_num"]).agg("count")
    # -------------------------------------------------------------------------------------------------------------
    #
    df_age_group_count = df.groupby("intressent_id").agg("mean").groupby("min_age").agg("count")
    #
    id_to_min_age = df.groupby("intressent_id").agg(
        lambda x: list(x)[0])[["min_age", "gender"]].to_dict(orient="index")
    #
    buckets = [defaultdict(set) for i in range(len(bucket_ranges))]
    #
    bucket_genders = [{"F": 0, "M": 0} for i in range(len(bucket_ranges))]
    #
    # first pass for gender balance
    for iid, d in id_to_min_age.items():
        min_age, gender = d.values()
        i = int((min_age-start)//5)
        bucket_size = len(buckets[i]["F"]) + len(buckets[i]["M"])
        if bucket_size < max_bucket_size \
        and bucket_genders[i][gender] < max_bucket_size//2:
            # buckets[i].add(iid)
            buckets[i][gender].add(iid)
            bucket_genders[i][gender] += 1
    #
    # second pass to fill buckets (some buckets otherwise don't have 4 speakers)
    for iid, d in id_to_min_age.items():
        min_age, gender = d.values()
        i = int((min_age-start)//5)
        bucket_size = len(buckets[i]["F"]) + len(buckets[i]["M"])
        if bucket_size < max_bucket_size and iid not in buckets[i][gender]:
            buckets[i][gender].add(iid)
            bucket_genders[i][gender] += 1
    #
    for i, bucket in enumerate(buckets):
        bucket_size = len(buckets[i]["F"]) + len(buckets[i]["M"])
        if bucket_size < 4:
            buckets[i] = {"F": set(), "M": set()}
            #
    return buckets


def get_test_df(df):
    # dataframe for speakers with 10+ years of speaking experience, 
    # and 3+ speeches every year they were active
    #
    print("Calculating over 10 and over 15 years speakers")
    df["first_debate"] = df.groupby(["intressent_id"])["debatedate"].transform(min)
    df["last_debate"] = df.groupby(["intressent_id"])["debatedate"].transform(max)
    #
    df["over_10"] = df[["first_debate", "last_debate"]].apply(
        lambda x: (x.last_debate - x.first_debate).days >= 365 * 10, axis=1)
    df_filt = df[(df["over_10"] == True)]
    #
    df_filt = df_filt.drop(["first_debate", "last_debate"], axis=1)
    # get only those speakers who spoke all 10+ years they were active
    df_filt["min_age"] = df_filt.groupby(["intressent_id"])["age"].transform("min")
    df_filt["max_age"] = df_filt.groupby(["intressent_id"])["age"].transform("max")
    df_filt["age_range"] = df_filt[["min_age", "max_age"]].apply(
        lambda x: set(range(int(x.min_age), int(x.max_age)+1)), axis=1)
    #
    sp_to_range = df_filt.groupby(["intressent_id"])["age"].apply(set).to_dict()
    df_filt["actual_age_range"] = df_filt["intressent_id"].apply(
        lambda x: sp_to_range[x])
    df_filt["spoke_all_years"] = df_filt[["actual_age_range", "age_range"]].apply(
        lambda x: x.actual_age_range == x.age_range, axis=1)
    #
    df_filt = df_filt[df_filt["spoke_all_years"] == True]
    #
    df_filt = df_filt.drop(["spoke_all_years", "min_age", "max_age", "age_range", "actual_age_range"], axis=1)
    #
    df = df_filt.reset_index(drop=True)
    return df


def get_train_df(df, df_buck_reduce):
    speakers_to_ignore = set(df_buck_reduce.intressent_id.tolist())
    df_train = df[df.intressent_id.apply(lambda x: x not in speakers_to_ignore)].reset_index(drop=True)
    #
    return df_train


def get_train_dev_ids(buckets, train_prop):
    train_ids = set()
    dev_ids = set()
    #
    for bucket in buckets:
        for gender in "FM":
            if len(bucket[gender]) < 4:
                train_ids.update(bucket[gender])
            else:
                split = int(len(bucket[gender])*train_prop)
                train_ids.update(list(bucket[gender])[:split])
                dev_ids.update(list(bucket[gender])[split:])
    return train_ids, dev_ids

# set up file paths ------------------------------------------------
os.chdir("..")
metadata = os.path.join(os.getcwd(), "metadata")
path_ts = os.path.join(metadata, "all_speeches_ts_downsize.parquet")
path_all = os.path.join(metadata, "riksdagen_speeches_with_ages.parquet")
path_test_bucketed = os.path.join(metadata, "bucketed_speeches.parquet")
path_train_bucketed = os.path.join(metadata, "bucketed_training_speeches.parquet")
path_dev_bucketed = os.path.join(metadata, "bucketed_dev_speeches.parquet")
os.chdir("scripts")

# open and merge dfs, filter for quality ---------------------------
df_ts = pd.read_parquet(path_ts)
df_all = pd.read_parquet(path_all)

cols = [col for col in df_all.columns if col not in df_ts.columns] + ["dokid_anfnummer"]

df = pd.merge(df_ts, df_all[cols], how="left", on=["dokid_anfnummer"])
df = quality_filter(df)

# create test data -------------------------------------------------
df_buck = get_test_df(df)

buckets = select_ids(df_buck, 4)

ids_to_use = set(reduce(lambda x, y: x.union(y), map(lambda x: x["F"].union(x["M"]), buckets)))
df_buck_reduce = df_buck[df_buck.intressent_id.apply(lambda x: x in ids_to_use)]
df_buck_reduce.to_parquet(path_test_bucketed, index=False)

# create training data ---------------------------------------------
df_train = get_train_df(df, df_buck_reduce)

buckets = select_ids(df_train, 20)

train_ids, dev_ids = get_train_dev_ids(buckets, 0.8)

df_train_reduce = df_train[df_train["intressent_id"].apply(lambda x: x in train_ids)].reset_index(drop=True)
df_train_reduce.to_parquet(path_train_bucketed, index=False)

df_dev_reduce = df_train[df_train["intressent_id"].apply(lambda x: x in dev_ids)].reset_index(drop=True)
df_dev_reduce.to_parquet(path_dev_bucketed, index=False)