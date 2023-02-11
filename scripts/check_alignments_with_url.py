import random
import pandas as pd
import os
import numpy as np

#---------------------------------------------------------------------
# data paths
thesis = os.getcwd()
while thesis.split("\\")[-1] != "masters_thesis":
    os.chdir("..")
    thesis = os.getcwd()

data_path = os.path.join(thesis, "metadata")

riksdagen = os.path.join(data_path, "riksdagen_speeches.parquet")
timestamp_path = os.path.join(data_path, "df_timestamp.parquet")
filtered_path = os.path.join(data_path, "filtered_speeches.parquet")

save_path = os.path.join(data_path, "riksdagen_speeches_with_ages.parquet")
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# opening dataframes
print("Opening dataframes")
##df = pd.read_parquet(riksdagen)
##df_meta = pd.read_csv(speaker_meta_path)
df_timestamp = pd.read_parquet(timestamp_path)
df_filt = pd.read_parquet(filtered_path)
#---------------------------------------------------------------------

print("Getting debate timestamps")

df_filt["debateurl_timestamp"] = (
  "https://www.riksdagen.se/views/pages/embedpage.aspx?did="
  + df_filt["dokid"]
  + "&start="
  + df_filt["start_segment"].astype(str)
  + "&end="
  + (df_filt["end_segment"]).astype(str)
)

num_speeches = len(df_filt)

debates_checked = set()

while len(debates_checked) != 50:
    debate_row = random.randint(0, num_speeches-1)

    debate = df_filt.iloc[debate_row]

    dokid = debate["dokid"]
    anfnummer = debate["anforande_nummer"]
    speaker = debate["shortname"]
    text = debate["anftext"]

    anf_id = dokid+"_"+str(anfnummer)
    
    end = debate["end_segment"]

    if anf_id not in debates_checked and not np.isnan(end):
        debates_checked.add(anf_id)

        url = debate["debateurl_timestamp"]

        secs = str(int(end%60)).zfill(2)
        rest = end/60
        mins = str(int(rest%60)).zfill(2)
        hours = str(int(rest/60)).zfill(2)
        end_time = f"{hours}:{mins}:{secs}"

        print(url)
        print(end_time)
        print(speaker)
        print(text[:75])

        input(f"Processed {len(debates_checked)} debates")
        print()
