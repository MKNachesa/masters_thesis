import pandas as pd
import os
import matplotlib.pyplot as plt

#---------------------------------------------------------------------
# data paths
thesis = os.getcwd()
while thesis.split("\\")[-1] != "masters_thesis":
    os.chdir("..")
    thesis = os.getcwd()

# thesis = "C:/Users/mayan/Documents/Language Technology Uppsala/Thesis"
data_path = os.path.join(thesis, "metadata")#/data")

save_path = os.path.join(data_path, "riksdagen_speeches_with_ages.parquet")
filtered_path = os.path.join(data_path, "filtered_speeches.parquet")
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# opening dataframes

print("Opening dataframes")
df = pd.read_parquet(save_path)
df_filt = pd.read_parquet(filtered_path)
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# info for data section of thesis

def print_df_data(df, filt=False):
    min_year, max_year = min(df["debatedate"].tolist()),\
                         max(df["debatedate"].tolist())
    if filt:
        print("W/ filter")
    else:
        print("W/o filter")
    print(f"first debate: {min_year}, last debate: {max_year}")

    num_debates = len(set(df["dokid"]))
    num_speeches = len(df)
    print(f"Number of debates: {num_debates}")
    print(f"Number of speeches: {num_speeches}")

    num_speakers = len(set(df["shortname"]))
    print(f"Number of speakers: {num_speakers}")

    min_age, max_age = min(df[df["age"] != 0]["age"].tolist()),\
                         max(df["age"].tolist())
    print(f"youngest age: {min_age}, oldest age: {max_age}")

    df["debateyear"] = df["debatedate"].apply(lambda x: x.year)

    num_debs_per_year = df.groupby("debateyear").agg(["count"])["dokid"]
    mean_debs = num_debs_per_year.mean()[0]
    std_debs = num_debs_per_year.std()[0]
    print(f"Debates per year mean: {mean_debs:.0f}, std: {std_debs:.0f}")
    min_debs, min_year = num_debs_per_year.min()[0], num_debs_per_year.idxmin()[0]
    max_debs, max_year = num_debs_per_year.max()[0], num_debs_per_year.idxmax()[0]
    print(f"Least debates ({min_debs}) in {min_year}")
    print(f"Most debates ({max_debs}) in {max_year}")

    num_debs_per_speaker = df.groupby("shortname").agg(["count"])["dokid"]
    mean_speeches = num_debs_per_speaker.mean()[0]
    std_speeches = num_debs_per_speaker.std()[0]
    print(f"Mean speeches per speaker: {mean_speeches:.0f}, std: {std_speeches:.0f}")

    num_debs_per_speaker = df.groupby(["shortname", "debateyear"])\
        .agg(["count"])["dokid"]
    mean_speeches = num_debs_per_speaker.mean()[0]
    std_speeches = num_debs_per_speaker.std()[0]
    print(f"Mean speeches per speaker per year: {mean_speeches:.0f}, \
std: {std_speeches:.0f}")

##    mean_speech_length = df.groupby("duration_segment")\
##        .agg(["mean"])["dokid"]
    mean_length = df["duration_segment"].mean()
    std_length = df["duration_segment"].std()
    print(f"Mean speech_length: {mean_length:.0f}, \
std: {std_length:.0f}")

print_df_data(df)
print()
print_df_data(df_filt, True)
#---------------------------------------------------------------------
