import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import defaultdict

#---------------------------------------------------------------------
# data paths
thesis = os.getcwd()
while thesis.split("\\")[-1] != "masters_thesis":
    os.chdir("..")
    thesis = os.getcwd()

# thesis = "C:/Users/mayan/Documents/Language Technology Uppsala/Thesis"
data_path = os.path.join(thesis, "metadata")#/data")

riksdagen = os.path.join(data_path, "riksdagen_speeches.parquet")
speaker_meta_path = os.path.join(data_path, "person.csv")
timestamp_path = os.path.join(data_path, "df_timestamp.parquet")

save_path = os.path.join(data_path, "riksdagen_speeches_with_ages.parquet")
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# opening dataframes
print("Opening dataframes")
df = pd.read_parquet(riksdagen)
df_meta = pd.read_csv(speaker_meta_path)
df_timestamp = pd.read_parquet(timestamp_path)
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# preprocessing speakers by
#   - lowercasing the name in speeches file ("text_lower")
#   - removing duplicate rows
#   - finding their name w/o pre- and postfixes in the metafile
#     and adding that in a new column ("shortname")
#       - this will omit a few speakers but that is a chance I'm willing to take

# lowercasing long names
print("Preprocessing speakers")

df["text_lower"] = df["text"].apply(lambda x: x.lower())

# remove duplicate rows
dup_speakers = [False]
for i, row in df[1:].iterrows():
    cur_debate = row["dokid"]
    prev_debate = df.iloc[i-1]["dokid"]
    cur_speaker = row["text_lower"]
    prev_speaker = df.iloc[i-1]["text_lower"]
    if cur_debate==prev_debate and cur_speaker==prev_speaker:
        dup_speakers.append(True)
    else:
        dup_speakers.append(False)
df["dup_speaker"] = dup_speakers
df = df[df["dup_speaker"] == False].reset_index()

# finding names of speakers w/o pre- and postfixes
df_meta["full_name"] = df_meta.apply(
    lambda x: " ".join([x.Förnamn, x.Efternamn]), axis=1)

longnames = set(df["text_lower"].tolist())
shortnames = set(df_meta["full_name"].str.lower().tolist())

long2shortname = dict()
short2longname = defaultdict(list)
for i, name in enumerate(shortnames):
    to_remove = []
    for longname in longnames:
        if name in longname:
            to_remove.append(longname)
            short2longname[name].append(longname)
            long2shortname[longname] = name
    for longname in to_remove:
        longnames.remove(longname)

df["shortname"] = df["text_lower"].apply(lambda x: long2shortname.get(x, None))

total_num_speakers = len(set(df["shortname"].to_list()))
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# get birthdays and age at time of debate

name_to_birthyear = dict(zip(df_meta["full_name"].str.lower(),
                             df_meta["Född"].astype(int)))

df["birthyear"] = df["shortname"].apply(lambda x:
                                        name_to_birthyear.get(x, None))

df["age"] = df[["birthyear", "debatedate"]].apply(lambda x:
                    x.debatedate.year - x.birthyear, axis=1)
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# getting speakers who were active for more than 10 and 15 years
# and saving dokid + anforande_nummer for 10+ to a parquet file

print("Calculating over 10 and over 15 years speakers")
over_15 = set()
over_10 = set()
for speaker in set(df["shortname"].to_list()):
    if speaker == None:
        continue
    sp_df = df[df["shortname"] == speaker]
    dates = sorted(sp_df["debatedate"].to_list())
    if (dates[-1] - dates[0]).days >= 365 * 15:
        over_15.add(speaker)
        over_10.add(speaker)
    elif (dates[-1] - dates[0]).days >= 365 * 10:
        over_10.add(speaker)

df["over_10"] = df["shortname"].apply(lambda x: x in over_10)
df["over_15"] = df["shortname"].apply(lambda x: x in over_15)

df_10 = df[df["over_10"] == True].reset_index()
save_dokid = os.path.join(data_path, "dokid_anfnum_over10_speeches.parquet")
df_10[["dokid", "anforande_nummer"]].to_parquet(save_dokid, index=False)
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# check debate timestamps
df_timestamp["debateurl_timestamp"] = (

"https://www.riksdagen.se/views/pages/embedpage.aspx?did="

+ df_timestamp["dokid"]

+ "&start="

+ df_timestamp["start_text_time"].astype(str)

+ "&end="

+ (df_timestamp["end_text_time"]).astype(str)

)

#---------------------------------------------------------------------

#---------------------------------------------------------------------
# save accumulated data to new file
print("Saving everything to new file")
df.to_parquet(save_path, index=False)

#---------------------------------------------------------------------

#---------------------------------------------------------------------
if False:
    deb_durs = df_10.groupby(["text_lower"], as_index=False)["duration"].sum()

    # bleu scores are not better for later years
    df_10["debateyear"] = df_10["debatedate"].apply(lambda x: x.year)
    df_10.groupby(["debateyear"], as_index=False)["bleu_score"].mean()

    print("Calculating debates overlap")
    overlaps = []
    for i, row in df[:-1].iterrows():
        id_cur = row["dokid"]
        id_next = df.iloc[i+1]["dokid"]
        start_next = df.iloc[i+1]["start"]
        end_cur = row["end"]
        if id_cur != id_next:
            overlaps.append(0)
            continue
        overlaps.append(max(0, end_cur - start_next))

    overlaps.append(0)
    df["overlap with next"] = overlaps

    overlaps = []
    for i, row in df[:-1].iterrows():
        id_cur = row["dokid"]
        id_next = df.iloc[i+1]["dokid"]
        start_next = df.iloc[i+1]["start_text_time"]
        end_cur = row["end_text_time"]
        if id_cur != id_next:
            overlaps.append(0)
            continue
        overlaps.append(max(0, end_cur - start_next))

    overlaps.append(0)
    df["overlap text with next"] = overlaps

    print("Calculating speeches to exclude if last")
    include = []
    for i, row in df[:-1].iterrows():
        cur_dokid = row["dokid"]
        next_dokid = df.iloc[i+1]["dokid"]
        if cur_dokid != next_dokid:
            include.append(False)
        else:
            include.append(True)

            
    include.append(False)
    df["include"] = include

    #---------------------------------------------------------------------
    # histograms n such

    # show overlaps with next based on text
    plt.hist(df["overlap text with next"][(df["overlap text with next"] <= 300) &
                                          (0 < df["overlap text with next"])])
    plt.show()

    # show overlaps with next based on text for 10+ years speaking
    plt.hist(df_10["overlap text with next"][(df_10["overlap text with next"] <= 300) &
                                          (0 < df_10["overlap text with next"])])
    plt.show()


    # show overlaps with next based on annotation
    plt.hist(df["overlap with next"][(df["overlap with next"] <= 300) &
                                          (0 < df["overlap with next"])])
    plt.show()

    # show overlaps with next based on annotation for 10+ years speaking

    plt.hist(df_10["overlap with next"][(df_10["overlap with next"] <= 300) &
                                          (0 < df_10["overlap with next"])])
    plt.show()

    #---------------------------------------------------------------------
