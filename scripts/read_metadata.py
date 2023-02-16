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

if False:
    # deduplicate rows
    print("Deduplicating full metadata and diarised metadata")
    dup_speeches = [False]
    duplicate_ids = set()
    ids_seen = set(df.iloc[0]["dokid_anfnummer"])
    for i, row in df[1:].iterrows():
        cur_debate = row["dokid_anfnummer"]
        if cur_debate in ids_seen:
            dup_speeches.append(True)
            df.loc[i-1, "text"] = "UNK"
            duplicate_ids.add(cur_debate)
        else:
            dup_speeches.append(False)
        ids_seen.add(cur_debate)
    df["dup_speech"] = dup_speeches
    df = df[df["dup_speech"] == False].reset_index(drop=True)

    diarise_ids = set(df_diarise["dokid_anfnummer"].tolist())
    df["in_diarise"] = df["dokid_anfnummer"].apply(
        lambda x: x in diarise_ids)
    df = df[df["in_diarise"] == True].reset_index(drop=True)

    # remove duplicate columns
    df.set_index("dokid_anfnummer", inplace=True)
    df_diarise.set_index("dokid_anfnummer", inplace=True)

    ##df_diarise["dokid_anfnummer2"] = df_diarise["dokid_anfnummer"]
    ##df_diarise["dokid2"] = df_diarise["dokid"]

    ##df = pd.concat([df, df_diarise], axis=1)
    df = df.join(df_diarise)

    df.reset_index(inplace=True)
    df_diarise.reset_index(inplace=True)

df_diarise_cols = df_diarise.columns.tolist()
for col in df_diarise_cols:
    if col in df.columns and col not in ["dokid", "anforande_nummer"]:
        df_diarise = df_diarise.drop(col, axis=1)

# Other way to join df and df_diarise
df = pd.merge(df, df_diarise, how="left", on=["dokid", "anforande_nummer"])

if False:
    # get rid of short debates with bad diarisation
    df = df[((df["length_ratio"] >= 0.7)
             & (df["length_ratio"] <= 1.3))
             & (df["duration_segment"] >= 20)].reset_index(drop=True)

#---------------------------------------------------------------------

#---------------------------------------------------------------------
# preprocessing speakers by
#   - lowercasing the name in speeches file ("text_lower")
#   - finding their name w/o pre- and postfixes in the metafile
#     and adding that in a new column ("shortname")
#       - this will omit a few speakers but that is a chance I'm willing to take

# lowercasing long names
print("Preprocessing speakers")

df["text_lower"] = df["text"].apply(lambda x: x.lower())

### find duplicate speeches; assumes first speaker is correct
##print("Marking duplicate speakers as <unk>")
##dup_speeches = [False]
##duplicate_ids = set()
##ids_seen = set(df.iloc[0]["dokid_anfnummer"])
##for i, row in df[1:].iterrows():
##    cur_debate = row["dokid_anfnummer"]
##    if cur_debate in ids_seen:
##        dup_speeches.append(True)
##        df.loc[i-1, "text_lower"] = "unk"
##        duplicate_ids.add(cur_debate)
##    else:
##        dup_speeches.append(False)
##    ids_seen.add(cur_debate)
##df["dup_speech"] = dup_speeches

if False:
    # take care of UNK speakers
    print("Taking care of UNK speakers")
    dokids = set(df["dokid"].tolist())
    print(len(dokids))
    for j, dokid in enumerate(dokids):
        if (j+1) % 500 == 0:
            print(j+1)
        label_to_text = dict()
        i_to_proc = []

        for i, row in df[df["dokid"] == dokid].iterrows():
            text = row["text_lower"]
            label = row["label"]
            if text != "unk":
                label_to_text[label] = text
            else:
                i_to_proc.append(i)

        for i in i_to_proc:
            label = df.iloc[i]["label"]
            df.at[i, "text_lower"] = label_to_text.get(label, "unk")

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

# marking duplicate anforanden
df["count_dokid_anfnummer"] = df.groupby(
    "dokid_anfnummer")["dokid_anfnummer"].transform("count")

# combining duplicate speech names
print("Combining duplicate speech names")
did_anfn_to_names = defaultdict(set)
for i, row in df.iterrows():
    name = row["shortname"]
    if name == None:
        name = "unk"
    dokid_anfnummer = row["dokid_anfnummer"]
    did_anfn_to_names[dokid_anfnummer].add(name)
    
df["combname"] = df["dokid_anfnummer"].apply(
    lambda x: " - ".join(did_anfn_to_names[x]))
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# get birthdays and age at time of debate

name_to_birthyear = dict(zip(df_meta["full_name"].str.lower(),
                             df_meta["Född"].astype(int)))

df["birthyear"] = df["shortname"].apply(lambda x:
                                        name_to_birthyear.get(x, 0)).astype(int)

df["age"] = df[["birthyear", "debatedate"]].apply(lambda x:
                    x.debatedate.year - x.birthyear
                    if x.birthyear > 0 else 0, axis=1).astype(int)
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

df_10 = df[df["over_10"] == True].reset_index(drop=True)
save_dokid = os.path.join(data_path, "dokid_anfnum_over10_speeches.parquet")
df_10[["dokid", "anforande_nummer"]].to_parquet(save_dokid, index=False)
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
# get speaker speaking ranges. I.e., did they speak in all the years between
# their first and last debate, or only some?

##df_filt["min_age"] = df_filt.groupby(["shortname"])["age"].transform("min")
##df_filt["max_age"] = df_filt.groupby(["shortname"])["age"].transform("max")
##df_filt["age_range"] = df_filt[["min_age", "max_age"]].apply(
##    lambda x: set(range(int(x.min_age), int(x.max_age)+1)), axis=1)
##
##sp_to_range = df_filt.groupby(["shortname"])["age"].apply(set).to_dict()
##df_filt["actual_age_range"] = df_filt.apply(lambda x: sp_to_range[x])
##df_filt["spoke_all_years"] = df_filt[["actual_age_range", "age_range"]].apply(
##    lambda x: x.actual_age_range == x.age_range, axis=1)

#---------------------------------------------------------------------

#---------------------------------------------------------------------
# Create dataframe with only speakers who have a shortname and birthday

# get rid of speakers with no shortname, birthyear,
# and who spoke less than 10 years
df_filt = df[(~df["shortname"].isna()) & (df["birthyear"] != 0)
             & (df["over_10"] == True)]

# get rid of invalid audio files and duplicate speeches
df_filt = df_filt[(df_filt["count_dokid_anfnummer"] == 1)
                  & (df_filt["valid_audio"] == True)]

df_filt = df_filt[df_filt["nr_speech_segments"] == 1.0]

df_filt = df_filt[(df_filt["length_ratio"] > 0.7)
                  & (df_filt["length_ratio"] < 1.3)]

df_filt = df_filt[(df_filt["overlap_ratio"] > 0.7)
                  & (df_filt["overlap_ratio"] < 1.3)]

# exclude speeches where speakers mention themselves (probably wrong name)
df_filt = df_filt[df_filt[["anftext", "shortname"]].apply(
    lambda x: x.shortname not in x.anftext.lower(), axis=1)]

# only keep speeches at least 80 secs long
df_filt = df_filt[df_filt["duration_segment"] >= 80]

# get only those speakers who spoke all 10+ years they were active
df_filt["min_age"] = df_filt.groupby(["shortname"])["age"].transform("min")
df_filt["max_age"] = df_filt.groupby(["shortname"])["age"].transform("max")
df_filt["age_range"] = df_filt[["min_age", "max_age"]].apply(
    lambda x: set(range(int(x.min_age), int(x.max_age)+1)), axis=1)

sp_to_range = df_filt.groupby(["shortname"])["age"].apply(set).to_dict()
df_filt["actual_age_range"] = df_filt["shortname"].apply(
    lambda x: sp_to_range[x])
df_filt["spoke_all_years"] = df_filt[["actual_age_range", "age_range"]].apply(
    lambda x: x.actual_age_range == x.age_range, axis=1)

df_filt = df_filt[df_filt["spoke_all_years"] == True]

# only keep speakers who have at least 3 speeches every year
at_least_3 = df_filt.groupby(
    ["shortname", "age"]).agg("count")["party"].to_dict()
df_filt["at_least_3"] = df_filt[["shortname", "age"]].apply(
    lambda x: at_least_3[(x.shortname, x.age)] >= 3, axis=1)
df_filt = df_filt[df_filt["at_least_3"] == True]

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

#---------------------------------------------------------------------
if False:
    #--------------------------------------------------
    # check each speech's overlap with the next
    for i, row in df_filt[:-1].iterrows():
        start = row["start_segment"]
        end = row["end_segment"]
        dokid = row["dokid"]
        next_start = df_filt.loc[i+1]["start_segment"]
        next_end = df_filt.loc[i+1]["end_segment"]
        next_dokid = df_filt.loc[i+1]["dokid"]

        if end > next_start and dokid == next_dokid and start < next_end:
            overlap = end - next_start
            if overlap < 1:
                continue
            print(overlap)
            overlap_ratio = row["overlap_ratio"]
            next_overlap_ratio = df_filt.loc[i+1]["overlap_ratio"]
            print(i, overlap, overlap_ratio, next_overlap_ratio, "\n",
                  start, end, "\n", next_start, next_end)
            inp = input()
            if inp == "b":
                break

    #--------------------------------------------------
        
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
