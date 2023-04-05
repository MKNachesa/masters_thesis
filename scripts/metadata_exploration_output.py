import pandas as pd
import os
import matplotlib.pyplot as plt
from functools import reduce

#---------------------------------------------------------------------
# data paths
thesis = os.getcwd()
while thesis.split("\\")[-1] != "masters_thesis":
    os.chdir("..")
    thesis = os.getcwd()

# thesis = "C:/Users/mayan/Documents/Language Technology Uppsala/Thesis"
data_path = os.path.join(thesis, "metadata")#/data")

save_path = os.path.join(data_path, "riksdagen_speeches_with_ages.parquet")
# filtered_path = os.path.join(data_path, "all_speeches_ts.parquet") # oops this file doesn't exist on my computer
ts_path = os.path.join(data_path, "all_speeches_ts_downsize.parquet") # contains fewer columns than the previous file + missing some rows due to VP extraction
# filtered_path = os.path.join(data_path, "filtered_speeches.parquet")
# across_age_path = os.path.join(data_path, "within_speaker_across_age_comparisons.parquet")
# within_age_path = os.path.join(data_path, "within_speaker_within_age_comparisons.parquet")
# across_speaker_path = os.path.join(data_path, "across_speaker_comparisons.parquet")
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# info for data section of thesis
# def dokid_anfnummers_used():
#     NUM_PAIRS = 3
#     danfs_to_use = set()    # dokid_anforande_nummers
#     for i in range(1, NUM_PAIRS+1):
#         danfs_to_use.update(set(reduce(lambda x, y: list(x) + list(y), list(map(set, df_across_age[f"pair_{i}"].tolist())))))
#     for i in range(1, NUM_PAIRS+1):
#         danfs_to_use.update(set(reduce(lambda x, y: list(x) + list(y), list(map(set, df_within_age[f"pair_{i}"].tolist())))))
#     danfs_to_use.update(set(reduce(lambda x, y: list(x) + list(y), list(map(set, df_within_age["pair_1"].tolist())))))
#     return danfs_to_use


def quality_filter(df):
    # because the filtered file doesn't exist an sich, maybe it should tho? Would save me time 🤔
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
        ["intressent_id", "age"]).agg("count")["shortname"].to_dict()
    df_filt["at_least_3"] = df_filt[["intressent_id", "age"]].apply(
        lambda x: at_least_3[(x.intressent_id, x.age)] >= 3, axis=1)
    df_filt = df_filt[df_filt["at_least_3"] == True]
    #
    # reset index
    df = df_filt.reset_index(drop=True)
    return df



def filter_df(df, danfs_to_use):
    df = df.copy()
    df = df[df["dokid_anfnummer"].apply(lambda x: x in danfs_to_use)]
    return df


def print_df_data(df, filt=False):
    min_year, max_year = min(df["debatedate"].tolist()),\
                         max(df["debatedate"].tolist())
    if filt:
        print("W/ filter")
    else:
        print("W/o filter")
    print(f"first debate: {min_year}, last debate: {max_year}")
    #
    num_debates = len(set(df["dokid"]))
    num_speeches = len(df)
    print(f"Number of debates: {num_debates}")
    print(f"Number of speeches: {num_speeches}")
    #
    num_speakers = len(set(df["intressent_id"]))
    print(f"Number of speakers: {num_speakers}")
    #
    min_age, max_age = min(df[df["age"] != 0]["age"].tolist()),\
                         max(df["age"].tolist())
    print(f"youngest age: {min_age}, oldest age: {max_age}")
    #
    df["debateyear"] = df["debatedate"].apply(lambda x: x.year)
    #
    num_debs_per_year = df.groupby("debateyear").agg(["count"])["dokid"]
    mean_debs = num_debs_per_year.mean()[0]
    std_debs = num_debs_per_year.std()[0]
    print(f"Debates per year mean: {mean_debs:.0f}, std: {std_debs:.0f}")
    min_debs, min_year = num_debs_per_year.min()[0], num_debs_per_year.idxmin()[0]
    max_debs, max_year = num_debs_per_year.max()[0], num_debs_per_year.idxmax()[0]
    print(f"Least debates ({min_debs}) in {min_year}")
    print(f"Most debates ({max_debs}) in {max_year}")
    #
    num_debs_per_speaker = df.groupby("intressent_id").agg(["count"])["dokid"]
    mean_speeches = num_debs_per_speaker.mean()[0]
    std_speeches = num_debs_per_speaker.std()[0]
    print(f"Mean speeches per speaker: {mean_speeches:.0f}, std: {std_speeches:.0f}")
    #
    num_debs_per_speaker = df.groupby(["intressent_id", "debateyear"])\
        .agg(["count"])["dokid"]
    mean_speeches = num_debs_per_speaker.mean()[0]
    std_speeches = num_debs_per_speaker.std()[0]
    print(f"Mean speeches per speaker per year: {mean_speeches:.0f}, \
std: {std_speeches:.0f}")
    #
##    mean_speech_length = df.groupby("duration_segment")\
##        .agg(["mean"])["dokid"]
    mean_length = df["duration_segment"].mean()
    std_length = df["duration_segment"].std()
    print(f"Mean speech_length: {mean_length:.0f}, \
std: {std_length:.0f}")
    #
# doesn't work in this file for some reason
##    starting_speaking_ages = sorted(list(set(
##        df.groupby("shortname").agg("min")["age"].tolist())))
##    print(f"Starting ages of speakers: \
##{', '.join(list(map(str, starting_speaking_ages)))}")
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# opening dataframes

print("Opening dataframes")
df = pd.read_parquet(save_path)
# df_filt = pd.read_parquet(filtered_path)  # the file in "filtered_path" doesn't exist on my computer
df_filt = df[df.duration_segment>=80].reset_index(drop=True)
df_ts = pd.read_parquet(ts_path)
df_ts = pd.merge(df_ts, df_filt[["debatedate", "dokid", "anforande_nummer", "age", "intressent_id", 
                                 "duration_segment", "shortname", "birthyear", "start_segment", 
                                 "count_dokid_anfnummer", "valid_audio", "nr_speech_segments",
                                 "length_ratio", "overlap_ratio", "anftext"]], 
                 how="left", on=["dokid", "anforande_nummer"]).drop_duplicates(["dokid", "anforande_nummer"]).reset_index(drop=True)
df_ts = quality_filter(df_ts)
# df_across_age = pd.read_parquet(across_age_path)
# df_within_age = pd.read_parquet(within_age_path)
# df_across_speaker = pd.read_parquet(across_speaker_path)
#---------------------------------------------------------------------

# danfs_to_use = dokid_anfnummers_used()
# df_bucket = filter_df(df, danfs_to_use)

print_df_data(df)
print()
# print_df_data(df_filt, True)
# print()
print_df_data(df_ts, True)
# print()
# print_df_data(df_bucket, True)
#---------------------------------------------------------------------
