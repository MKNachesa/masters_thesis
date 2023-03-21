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
#   - getting their gender

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

id_to_gender = df_meta.groupby("Id").agg(lambda x: "F" if list(x)[0] == "kvinna" else "M")["Kön"].to_dict()
df["gender"] = df.intressent_id.apply(lambda x: id_to_gender[x] if x in id_to_gender else None)
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
# save accumulated data to new file

if True:
    print("Saving everything to new file")
    df.to_parquet(save_path, index=False)
    # df_filt.to_parquet(filtered_path, index=False)
    # df_downsize.to_parquet(downsize_path, index=False)

#---------------------------------------------------------------------
