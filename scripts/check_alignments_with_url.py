import random
import pandas as pd
import os
import numpy as np

#---------------------------------------------------------------------
# data paths
same_folder = "n"
print_message = True

if print_message:
    print("""This script reads the data file \
'downsize_filtered_speeches.parquet'.
The script and the data need to be in either of two configurations:
  - this script and the data file are in the same folder
  - this script and this file are part of the following directory structure:
    - masters_thesis
      |_ metadata
         |_ data_file
      |_ scripts
         |_ this script

Please make sure you follow either of these two configurations.

If you wish to suppress this message, open this script and set the variable
'same_folder' to either 'y' or 'n' and 'print_message' to False.
""")

    same_folder = input("Is this script in the same folder as the data? [y/n] ")
    print()
    while same_folder not in {"y", "n"}:
        print(f"{same_folder} is not valid input")
        same_folder = input(
            "Is this script in the same folder as the data? [y/n] ")
        print()

assert same_folder in {"y", "n"}

if same_folder == "y":
##    filtered_path = "filtered_speeches.parquet"
    downsize_path = "downsize_filtered_speeches.parquet"

else:
    thesis = os.getcwd()
    while thesis.split("\\")[-1] != "masters_thesis":
        os.chdir("..")
        thesis = os.getcwd()

    data_path = os.path.join(thesis, "metadata")

##    filtered_path = os.path.join(data_path, "filtered_speeches.parquet")
    downsize_path = os.path.join(data_path,
                                 "downsize_filtered_speeches.parquet")
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# opening dataframes
print("Opening dataframes\n")
##df_filt = pd.read_parquet(filtered_path)
df_downsize = pd.read_parquet(downsize_path)
#---------------------------------------------------------------------

num_speeches = len(df_downsize)

debates_checked = set()
num_of_speeches_to_check = 50

print("""Checking {num_of_speeches_to_check} speeches.
After each speech, press any key to continue
""")

while len(debates_checked) != num_of_speeches_to_check:
    debate_row = random.randint(0, num_speeches-1)

    debate = df_downsize.iloc[debate_row]

    anf_id = debate["dokid_anfnummer"]
    speaker = debate["shortname"]
    text = debate["anftext"]
    
    end = debate["end_segment"]

    if anf_id not in debates_checked and not np.isnan(end):
        debates_checked.add(anf_id)

        url = debate["debateurl_timestamp"]

        secs = str(int(end%60)).zfill(2)
        rest = end/60
        mins = str(int(rest%60)).zfill(2)
        hours = str(int(rest/60)).zfill(2)
        end_time = f"{hours}:{mins}:{secs}"

        print(f"speech id: {anf_id}")
        print(f"end time: {end_time}")
        print(f"speaker: {speaker}")
        print(f"text: {text[:70]}")
        print(f"url: {url}")

        input(f"Processed {len(debates_checked)} debates")
        print()
