import pandas as pd
import os

thesis = os.getcwd()
while thesis.split("\\")[-1] != "masters_thesis":
    os.chdir("..")
    thesis = os.getcwd()

metadata_path = os.path.join(thesis, "metadata")#/data")

full_df_path = os.path.join(metadata_path, "riksdagen_speeches_with_ages.parquet")
bucket_path = os.path.join(metadata_path, "bucketed_speeches.parquet")
train_path = os.path.join(metadata_path, "training_speeches.parquet")
speaker_meta_path = os.path.join(data_path, "person.csv")

df = pd.read_parquet(full_df_path)
df_bucket = pd.read_parquet(bucket_path)
df_meta = pd.read_csv(speaker_meta_path)

#-----------------------------------------------------------------------

speakers_used = set(df_bucket.intressent_id.tolist())
if "gender" not in df.columns:
    id_to_gender = df_meta.groupby("Id").agg(lambda x: "F" if list(x)[0] == "kvinna" else "M")["Kön"].to_dict()
    df["gender"] = df.intressent_id.apply(lambda x: id_to_gender[x] if x in id_to_gender else None)

df_filt = df[(~df["shortname"].isna()) & (df["birthyear"] != 0)
             & df.intressent_id.apply(lambda x: x not in speakers_used)]

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
       'over_15', 'debateurl_timestamp', "gender"]]

# reset index
df_filt = df_filt.reset_index(drop=True)

# add info ---------------------------------------------------------------------
min_age = df_filt.age.min()
df_filt["bucket"] = df_filt.age.apply(lambda x: f"{min_age+((x-min_age)//5)*5}-{min_age+((x-min_age)//5)*5+5}")

# print number of speeches per age bucket and gender
buckets = sorted(list(set(df_filt.bucket.tolist())))
groups = df_filt.groupby(["gender", "bucket"]).agg("count").dokid

if True:
    print(f"{'bucket':<6}\t{'F count':>7}\t{'M count':>7}")
    for bucket in buckets:
        try:
            m_count = groups[("M", bucket)]
        except:
            m_count = 0
        try:
            f_count = groups[("F", bucket)]
        except:
            f_count = 0
        print(f"{bucket:<6}\t{f_count:>7}\t{m_count:>7}")