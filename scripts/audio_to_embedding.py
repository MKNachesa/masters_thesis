import pandas as pd
import os
import pickle as pkl
import nemo.collections.asr as nemo_asr
from tqdm import tqdm
from pathlib import Path

script_dir = os.getcwd()
os.chdir("..")
thesis_dir = os.getcwd()
audio_dir = "/data/datasets/riksdagen_anforanden/data/audio"
os.chdir(script_dir)

data_dir = os.path.join(thesis_dir, "data")
if "data" not in next(os.walk(thesis_dir))[1]:
    os.mkdir(data_dir)

metadata_dir = os.path.join(thesis_dir, "metadata")
# bucket_file = os.path.join(metadata_dir, "bucketed_speeches.parquet")
bucket_file = os.path.join(metadata_dir, "all_speeches_ts_downsize.parquet")

df = pd.read_parquet(bucket_file)
speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')

danfs_to_ignore = set()

# for dur in [None]:#, 60, 30, 10, 5, 3, 1]:#
for dur in [None, 60, 30, 10, 5, 3, 1]:    #None, 60, , 
    dur_str = dur if dur else "full"
    df[f"filename_anforande_audio_{dur_str}"] = df[["filename", f"timestamps_{dur_str}"]].apply(
        lambda x: os.path.join(audio_dir, f"{x.filename.strip('.wav')}_{x[f'timestamps_{dur_str}'][0]}_{x[f'timestamps_{dur_str}'][1]}.wav"), axis=1)

    df_groups = df.groupby("dokid")
    df_groups = df_groups[["dokid", "anforande_nummer", "filename", f"timestamps_{dur_str}", "dokid_anfnummer", f"filename_anforande_audio_{dur_str}"]]
    df_list = [df_groups.get_group(x) for x in df_groups.groups]  # list of dfs, one for each dokid

    files_processed = 0
    with tqdm(df_list) as dfs:
        for sub_df in dfs:
            dok_to_emb = dict()
            dokid = sub_df.dokid.iloc[0]
            dfs.set_postfix(dokid=dokid, dur=dur_str)
            dur_dir = os.path.join(data_dir, f"{dur_str}")
            save_file = os.path.join(dur_dir, f"emb_{dokid}.pkl")
            # if Path(save_file).is_file() and os.path.getsize(save_file) > 0:
            #     continue
            files_processed += 1
            for i, row in sub_df.iterrows():
                danf = row.dokid_anfnummer
                if danf in danfs_to_ignore: # ignore speeches that are too big
                    continue

                # if (i+1) % 20 == 0:
                #     print(f"loop {i+1:>4}/{len(sub_df)}", end="\r", flush=True)
                file_path = row[f"filename_anforande_audio_{dur_str}"]
                dok = row.dokid_anfnummer
                try:
                    emb = speaker_model.get_embedding(file_path).cpu()
                except:
                    # probably the full speech was too long
                    danfs_to_ignore.add(row.dokid_anfnummer)
                    continue
                dok_to_emb[dok] = emb
            if f"{dur_str}" not in next(os.walk(data_dir))[1]:
                os.mkdir(dur_dir)
            f = open(save_file, "wb")
            pkl.dump(dok_to_emb, f)
            f.close()
        if not files_processed:
            print("No new files created")
        else:
            print(f"Created {files_processed} files")

df = df[df.dokid_anfnummer.apply(lambda x: x not in danfs_to_ignore)].reset_index(drop=True)
df.to_parquet("../metadata/all_speeches_ts_downsize.parquet", index=False)