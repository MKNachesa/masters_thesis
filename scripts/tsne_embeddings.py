import os
import pickle as pkl
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

NUM_PAIRS = 3
os.chdir("../metadata")
metadata_dir = os.getcwd()
os.chdir("../results")
results_dir = os.getcwd()
os.chdir("../data")
data_dir = os.getcwd()
path_bucket = os.path.join(metadata_dir, "bucketed_speeches.parquet")

# df_buck = pd.read_parquet(path_bucket)
# buck_danfs = set(df_buck.dokid_anfnummer.tolist())
# dokids_to_process = list(map(lambda x: x.split("_")[0], buck_danfs))
# dfs = dict()

# # tsne all speech lengths combined
# direcs = ["1", "3", "5", "10", "30", "60", "full"]
# df = pd.DataFrame(columns=["dokid_anfnummer", "emb"])
# for direc in direcs:
#     print(f"Processing {direc}")
#     os.chdir(direc)
#     for dokid in dokids_to_process:
#     # for dokid in os.listdir():
#         with open(f"emb_{dokid}.pkl", "rb") as infile:
#             f = pkl.load(infile)
#         tmp_df = pd.DataFrame(f.items(), columns=["dokid_anfnummer", "emb"])
#         df = pd.concat([df, tmp_df])
#     os.chdir("..")
# #                        
# df["in_buck"] = df.dokid_anfnummer.apply(lambda x: x in buck_danfs)
# #                                                          
# df = df[df["in_buck"]==True].reset_index(drop=True)                                        
# df_mg = pd.merge(df, df_buck, how="left", on=["dokid_anfnummer"])
# df_mg = df_mg.drop("in_buck", axis=1)
# df_mg["emb"] = df_mg["emb"].apply(lambda x: np.array(x[0]))
# #
# tsne = TSNE(n_components=2, verbose=1, random_state=123)
# z = tsne.fit_transform(np.array(df_mg.emb.tolist()))
# df_mg["comp-1"] = z[:,0]
# df_mg["comp-2"] = z[:,1]
# ids = set(df_mg.intressent_id.tolist())
# id_to_num = dict()
# for i, iid in enumerate(ids):
#     id_to_num[iid] = i

# df_mg["num"] = df_mg.intressent_id.apply(lambda x: id_to_num[x])
# df_mg["num_name"] = df_mg[["num", "shortname"]].apply(
#     lambda x: f"{x.num}_{x.shortname}", axis=1)

df_mg = pd.read_parquet(os.path.join(metadata_dir, "tsne_df.parquet"))
df_mg = df_mg.sort_values("intressent_id")
# df_mg = df_mg.sample(frac=1)

ids = set(df_mg.intressent_id.tolist())
ids = sorted(list(ids))
labels = ["B", "E", "A", "C", "D", "A", "F", "C", "D", "E", "D", "A", "C", "F", "C", "A", "B", "B", "B", "E"]
ids_to_labels = dict(zip(ids, labels))
hue = [ids_to_labels[id] for id in df_mg.intressent_id.tolist()]
# custom_palette = {'A': 'red', 'B': 'orange', 'C': 'yellow', 'D': 'green', 'E': 'blue', 'F': 'purple'}

ax = sns.scatterplot(x="comp-1", y="comp-2", hue=df_mg.intressent_id.tolist(),
            palette=sns.color_palette("hls", 20),
            data=df_mg)
# ax.set(title=f"embedding separation for all speech lengths")
sns.move_legend(ax, "center left", bbox_to_anchor=(1,0.5))
plt.savefig(os.path.join(results_dir, 
                         "T-SNE graphs",
                         f'T-SNE for all speeches at all speech lengths.png'), 
                         bbox_inches='tight', dpi=300)
plt.close()

# df_mg.to_parquet(os.path.join(metadata_dir, "tsne_df.parquet"))

# tsne all speech lengths
if False:
    direcs = ["1", "3", "5", "10", "30", "60", "full"]
    for direc in direcs:
        print(f"Processing {direc}")
        os.chdir(direc)
        df = pd.DataFrame(columns=["dokid_anfnummer", "emb"])
        for dokid in os.listdir():
            with open(dokid, "rb") as infile:
                f = pkl.load(infile)
            tmp_df = pd.DataFrame(f.items(), columns=["dokid_anfnummer", "emb"])
            df = pd.concat([df, tmp_df])
        #                                  
        df["in_buck"] = df.dokid_anfnummer.apply(lambda x: x in buck_danfs)
        #                                                          
        df = df[df["in_buck"]==True].reset_index(drop=True)                                        
        df_mg = pd.merge(df, df_buck, how="left", on=["dokid_anfnummer"])
        df_mg = df_mg.drop("in_buck", axis=1)
        df_mg["emb"] = df_mg["emb"].apply(lambda x: np.array(x[0]))
    #
        tsne = TSNE(n_components=2, verbose=1, random_state=123)
        z = tsne.fit_transform(np.array(df_mg.emb.tolist()))
        df_mg["comp-1"] = z[:,0]
        df_mg["comp-2"] = z[:,1]
        ids = set(df_mg.intressent_id.tolist())
        id_to_num = dict()
        for i, iid in enumerate(ids):
            id_to_num[iid] = i
        #
        df_mg["num"] = df_mg.intressent_id.apply(lambda x: id_to_num[x])
        df_mg["num_name"] = df_mg[["num", "shortname"]].apply(
            lambda x: f"{x.num}_{x.shortname}", axis=1)
        #
        dfs[direc] = df_mg
        os.chdir("..")
        print()

    for d in direcs:
        ax = sns.scatterplot(x="comp-1", y="comp-2", hue=dfs[d].num_name.tolist(),
                    palette=sns.color_palette("hls", 32),
                    data=dfs[d])
        ax.set(title=f"embedding separation for {d} (sec) speech length")
        sns.move_legend(ax, "center left", bbox_to_anchor=(1,0.5))
        plt.savefig(os.path.join(results_dir, f'T-SNE for all speeches at {d} speech length.png'), bbox_inches='tight')
        plt.close()


    # tsne with subset of speeches
    df = pd.read_parquet("../metadata/within_speaker_within_age_comparisons.parquet")
    danfs_to_use = set()

    for i in range(1, NUM_PAIRS+1):
        danfs_to_use.update(set(reduce(lambda x, y: list(x) + list(y), list(map(set, df[f"pair_{i}"].tolist())))))

    df_buck = df_buck[df_buck.dokid_anfnummer.apply(lambda x: x in danfs_to_use)].reset_index(drop=True)
    buck_danfs = set(df_buck.dokid_anfnummer.tolist())
    dfs = dict()

    for direc in direcs:
        print(f"Processing {direc}")
        os.chdir(direc)
        #
        df = pd.DataFrame(columns=["dokid_anfnummer", "emb"])
        ##for dokid in dokids_to_process:
        for dokid in os.listdir():
            with open(dokid, "rb") as infile:
                f = pkl.load(infile)
            tmp_df = pd.DataFrame(f.items(), columns=["dokid_anfnummer", "emb"])
            df = pd.concat([df, tmp_df])
        #                                  
        df["in_buck"] = df.dokid_anfnummer.apply(lambda x: x in buck_danfs)
        #                                                          
        df = df[df["in_buck"]==True].reset_index(drop=True)
        # df = df.reset_index(drop=True)
            #                                                        
        df_mg = pd.merge(df, df_buck, how="left", on=["dokid_anfnummer"])
        df_mg = df_mg.drop("in_buck", axis=1)
        df_mg["emb"] = df_mg["emb"].apply(lambda x: np.array(x[0]))
    #
        # where do the extra rows come from?
        # df_buck has fewer rows than df? euhh
        # df_mg = df_mg[~df_mg.intressent_id.isna()]
    #
        tsne = TSNE(n_components=2, verbose=1, random_state=123, perplexity=60)
        z = tsne.fit_transform(np.array(df_mg.emb.tolist()))
        df_mg["comp-1"] = z[:,0]
        df_mg["comp-2"] = z[:,1]
        ids = set(df_mg.intressent_id.tolist())
        id_to_num = dict()
        for i, iid in enumerate(ids):
            id_to_num[iid] = i
        #
        df_mg["num"] = df_mg.intressent_id.apply(lambda x: id_to_num[x])
        df_mg["num_name"] = df_mg[["num", "shortname"]].apply(
            lambda x: f"{x.num}_{x.shortname}", axis=1)
        #
        dfs[direc] = df_mg
        os.chdir("..")
        print()

    for d in direcs:
        ax = sns.scatterplot(x="comp-1", y="comp-2", hue=dfs[d].num_name.tolist(),
                    palette=sns.color_palette("hls", 32),
                    data=dfs[d])
        ax.set(title=f"embedding separation for {d} (sec) speech length")
        sns.move_legend(ax, "center left", bbox_to_anchor=(1,0.5))
        plt.savefig(os.path.join(results_dir, f'T-SNE for subset of speeches at {d} speech length.png'), bbox_inches='tight')
        plt.close()