import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import re

#---------------------------------------------------------------------
# data paths
thesis = os.getcwd()
while re.split(r"\\|/", os.getcwd())[-1] != "masters_thesis":
    os.chdir("..")
    thesis = os.getcwd()

# thesis = "C:/Users/mayan/Documents/Language Technology Uppsala/Thesis"
data_path = os.path.join(thesis, "metadata")#/data")

riksdagen = os.path.join(data_path, "riksdagen_speeches.parquet")
speaker_meta_path = os.path.join(data_path, "person.csv")
timestamp_path = os.path.join(data_path, "df_timestamp.parquet")
diarisation_path = os.path.join(data_path, "df_diarization.parquet")

save_path = os.path.join(data_path, "riksdagen_speeches_with_ages.parquet")
filtered_path = os.path.join(data_path, "filtered_speeches.parquet")
downsize_path = os.path.join(data_path, "downsize_filtered_speeches.parquet")
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# opening dataframes

print("Opening dataframes")
df = pd.read_parquet(riksdagen)
df_meta = pd.read_csv(speaker_meta_path)
df_timestamp = pd.read_parquet(timestamp_path)
df_diarise = pd.read_parquet(diarisation_path)
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# preprocess duplicate rows

df["dokid_anfnummer"] = df.apply(
    lambda x: f"{x.dokid}_{x.anforande_nummer}", axis=1)
df_diarise["dokid_anfnummer"] = df_diarise.apply(
    lambda x: f"{x.dokid}_{x.anforande_nummer}", axis=1)

df_diarise_cols = df_diarise.columns.tolist()
for col in df_diarise_cols:
    if col in df.columns and col not in ["dokid", "anforande_nummer"]:
        df_diarise = df_diarise.drop(col, axis=1)

# Other way to join df and df_diarise
df = pd.merge(df, df_diarise, how="left", on=["dokid", "anforande_nummer"])
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# preprocessing speakers by
#   - lowercasing the name in speeches file ("text_lower")
#   - finding their name w/o pre- and postfixes in the metafile
#     and adding that in a new column ("shortname")
#       - this will omit a few speakers but that is a chance I'm willing to take

# lowercasing long names
print("Preprocessing speakers")

# finding names of speakers w/o pre- and postfixes
df_meta["full_name"] = df_meta.apply(
    lambda x: " ".join([x.Förnamn, x.Efternamn]).lower(), axis=1)

# sometimes two speakers have the same name
# intressent_id is how one distinguishes them
# those without an intressent_id in the meta file are excluded
id_to_name = dict()

for i, row in df_meta[~df_meta.Id.isna()].iterrows():
    id_to_name[row.Id] = row.full_name

df["shortname"] = df["intressent_id"].apply(lambda x: id_to_name.get(x, None))

total_num_speakers = len(set(df["intressent_id"].to_list()))

# marking duplicate anforanden
df["count_dokid_anfnummer"] = df.groupby(
    "dokid_anfnummer")["dokid_anfnummer"].transform("count")
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# get birthdays and age at time of debate

id_to_birthyear = dict(zip(df_meta["Id"].str.lower(),
                             df_meta["Född"].astype(int)))

df["birthyear"] = df["intressent_id"].apply(lambda x:
    id_to_birthyear.get(x, 0)).astype(int)

df["age"] = df[["birthyear", "debatedate"]].apply(lambda x:
                    x.debatedate.year - x.birthyear
                    if x.birthyear > 0 else 0, axis=1).astype(int)
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# getting speakers who were active for more than 10 and 15 years
# and saving dokid + anforande_nummer for 10+ to a parquet file

print("Calculating over 10 and over 15 years speakers")
df["first_debate"] = df.groupby(["intressent_id"])["debatedate"].transform(min)
df["last_debate"] = df.groupby(["intressent_id"])["debatedate"].transform(max)

df["over_10"] = df[["first_debate", "last_debate"]].apply(
    lambda x: (x.last_debate - x.first_debate).days >= 365 * 10, axis=1)

df["over_15"] = df[["first_debate", "last_debate"]].apply(
    lambda x: (x.last_debate - x.first_debate).days >= 365 * 15, axis=1)

df_10 = df[df["over_10"] == True].reset_index(drop=True)
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# check debate timestamps
print("Getting debate timestamps")

df_timestamp["debateurl_timestamp"] = (
  "https://www.riksdagen.se/views/pages/embedpage.aspx?did="
  + df_timestamp["dokid"]
  + "&start="
  + df_timestamp["start_text_time"].astype(str)
  + "&end="
  + (df_timestamp["end_text_time"]).astype(str)
)

df["debateurl_timestamp"] = (
  "https://www.riksdagen.se/views/pages/embedpage.aspx?did="
  + df["dokid"]
  + "&start="
  + df["start_segment"].astype(str)
  + "&end="
  + (df["end_segment"]).astype(str)
)

#---------------------------------------------------------------------

#---------------------------------------------------------------------
# Create dataframe with only speakers who have a shortname and birthday

# get rid of speakers with no shortname, birthyear,
# and who spoke less than 10 years
# automatically gets rid of lines with no intressent_id
df_filt = df[(~df["shortname"].isna()) & (df["birthyear"] != 0)
             & (df["over_10"] == True)]

# deduplicate rows
df_filt = df_filt.drop_duplicates(
    ["dokid_anfnummer", "intressent_id", "start_segment"])

# get rid of invalid audio files and duplicate speeches
df_filt = df_filt[(df_filt["count_dokid_anfnummer"] == 1)
                  & (df_filt["valid_audio"] == True)
                  & (~df_filt["start_segment"].isna())]

# only keep speeches with 1 speech segment
df_filt = df_filt[df_filt["nr_speech_segments"] == 1.0]

# keep speeches within this length ratio
df_filt = df_filt[(df_filt["length_ratio"] > 0.7)
                  & (df_filt["length_ratio"] < 1.3)]

# keep speeches within this overlap ratio
df_filt = df_filt[(df_filt["overlap_ratio"] > 0.7)
                  & (df_filt["overlap_ratio"] < 1.3)]

# exclude speeches where speakers mention themselves (probably wrong name)
df_filt = df_filt[df_filt[["anftext", "shortname"]].apply(
    lambda x: x.shortname not in x.anftext.lower(), axis=1)]

# only keep speeches at least 80 secs long
df_filt = df_filt[df_filt["duration_segment"] >= 80]

# get only those speakers who spoke all 10+ years they were active
df_filt["min_age"] = df_filt.groupby(["intressent_id"])["age"].transform("min")
df_filt["max_age"] = df_filt.groupby(["intressent_id"])["age"].transform("max")
df_filt["age_range"] = df_filt[["min_age", "max_age"]].apply(
    lambda x: set(range(int(x.min_age), int(x.max_age)+1)), axis=1)

sp_to_range = df_filt.groupby(["intressent_id"])["age"].apply(set).to_dict()
df_filt["actual_age_range"] = df_filt["intressent_id"].apply(
    lambda x: sp_to_range[x])
df_filt["spoke_all_years"] = df_filt[["actual_age_range", "age_range"]].apply(
    lambda x: x.actual_age_range == x.age_range, axis=1)

df_filt = df_filt[df_filt["spoke_all_years"] == True]

# reset index
df_filt = df_filt.reset_index(drop=True)

# get rid of non-monotonically increasing speeches
print("Getting rid of non-monotonically following speeches")
dokids = set(df_filt["dokid"].tolist())
non_monotonic = set()
for i, dokid in enumerate(dokids):
    if (i+1) % 1000 == 0:
        print(f"Processed {i+1}/{len(dokids)} debates")
    mini_df = df_filt[df_filt["dokid"] == dokid]
    for i, row in mini_df[1:].iterrows():
        anf = row["anforande_nummer"]
        prev_anf = mini_df.loc[i-1]["anforande_nummer"]
        if prev_anf > anf:
            anf = row["dokid_anfnummer"]
            prev_anf = mini_df.loc[i-1]["dokid_anfnummer"]
            non_monotonic.add(prev_anf)
            non_monotonic.add(anf)

df_filt = df_filt[df_filt["dokid_anfnummer"].apply(lambda x: x not in non_monotonic)]

# only keep speakers who have at least 3 speeches every year
at_least_3 = df_filt.groupby(
    ["intressent_id", "age"]).agg("count")["party"].to_dict()
df_filt["at_least_3"] = df_filt[["intressent_id", "age"]].apply(
    lambda x: at_least_3[(x.intressent_id, x.age)] >= 3, axis=1)
df_filt = df_filt[df_filt["at_least_3"] == True]

# remove some irrelevant columns
df_filt = df_filt[['dokid', 'anforande_nummer', 'start', 'duration',
       'debateseconds', 'text', 'debatedate', 'url',
       'debateurl', 'id', 'audiofileurl', 'downloadfileurl',
       'anftext', 'filename', 'intressent_id', 'dokid_anfnummer', 'label',
       'start_segment', 'end_segment', 'duration_segment', 'end',
       'start_text_time', 'end_text_time', 'duration_text', 'duration_overlap',
       'overlap_ratio', 'length_ratio', 'shortname', 'birthyear', 'age',
       'over_15', 'debateurl_timestamp']]

# reset index
df_filt = df_filt.reset_index(drop=True)

# create smaller dataframe for putting on github and checking speeches
df_downsize = df_filt[["anftext", "dokid_anfnummer", "start_segment",
                    "end_segment", "shortname",
                    "debateurl_timestamp"]].reset_index(drop=True)
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# save accumulated data to new file

if True:
    print("Saving everything to new file")
    df.to_parquet(save_path, index=False)
    df_filt.to_parquet(filtered_path, index=False)
    df_downsize.to_parquet(downsize_path, index=False)

#---------------------------------------------------------------------
