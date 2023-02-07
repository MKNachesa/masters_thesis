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
##df = pd.read_parquet(riksdagen)
##df_meta = pd.read_csv(speaker_meta_path)
df_timestamp = pd.read_parquet(timestamp_path)
#---------------------------------------------------------------------

print("Getting debate timestamps")

df_timestamp["debateurl_timestamp"] = (
  "https://www.riksdagen.se/views/pages/embedpage.aspx?did="
  + df_timestamp["dokid"]
  + "&start="
  + df_timestamp["start_text_time"].astype(str)
  + "&end="
  + (df_timestamp["end_text_time"]).astype(str)
)

num_speeches = len(df_timestamp)

debates_checked = set()

while len(debates_checked) != 50:
    debate_row = random.randint(0, num_speeches-1)

    debate = df_timestamp.iloc[debate_row]

    dokid = debate["dokid"]
    anfnummer = debate["anforande_nummer"]

    anf_id = dokid+"_"+str(anfnummer)
    
    end = debate["end_text_time"]

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

        input(f"Processed {len(debates_checked)} debates")
        print()
