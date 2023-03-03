import os
import pickle as pkl
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

os.chdir("../metadata")
metadata_dir = os.getcwd()
os.chdir("../data")
data_dir = os.getcwd()
path_bucket = os.path.join(metadata_dir, "bucketed_speeches.parquet")

dfs = dict()

direcs = ["1", "3", "5", "10", "30", "60", "full"]
for direc in direcs:
    print(f"Processing {direc}")
    os.chdir(direc)

    ##dokids_to_process = {'GZ01CU14', 'GZ01CU18', 'GZ01CU27', 'GZ01NU24', 'GZ01NU3', 'H001AU11', 
    ##                     'H001CU10', 'H001FöU3', 'H201SoU14', 'H301UU12', 'H401AU9', 'H401CU10', 
    ##                     'H401TU4', 'H401TU6', 'H501MJU22', 'H501MJU24', 'H601TU11', 'H601TU5', 
    ##                     'H701TU7', 'H801TU16', 'H901TU1', 'H901UU15'}

    df = pd.DataFrame(columns=["dokid_anfnummer", "emb"])
    ##for dokid in dokids_to_process:
    for dokid in os.listdir():
##        if dokid == 'emb_GT01UbU1.pkl' or dokid == 'emb_GZ01FiU1.pkl':
##            continue
    ##    with open(f"emb_{dokid}.pkl", "rb") as infile:
        with open(dokid, "rb") as infile:
            f = pkl.load(infile)
        tmp_df = pd.DataFrame(f.items(), columns=["dokid_anfnummer", "emb"])
        df = pd.concat([df, tmp_df])

                           
    df_buck = pd.read_parquet(path_bucket)
                                                                  
    df_buck["in_df"] = df_buck.dokid_anfnummer.apply(lambda x: x in set(df.dokid_anfnummer.tolist()))
                                                                  
    df_buck = df_buck[df_buck["in_df"]==True].reset_index(drop=True)
    df = df.reset_index(drop=True)
                                                                  
    df_mg = pd.merge(df, df_buck, how="left", on=["dokid_anfnummer"])
    df_mg = df_mg.drop("in_df", axis=1)
    df_mg["emb"] = df_mg["emb"].apply(lambda x: np.array(x[0]))

    # where do the extra rows come from?
    # df_buck has fewer rows than df? euhh
    df_mg = df_mg[~df_mg.intressent_id.isna()]

    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(np.array(df_mg.emb.tolist()))
    df_mg["comp-1"] = z[:,0]
    df_mg["comp-2"] = z[:,1]
    ids = set(df_mg.intressent_id.tolist())
    id_to_num = dict()
    for i, iid in enumerate(ids):
        id_to_num[iid] = i

            
    df_mg["num"] = df_mg.intressent_id.apply(lambda x: id_to_num[x])
    df_mg["num_name"] = df_mg[["num", "shortname"]].apply(
        lambda x: f"{x.num}_{x.shortname}", axis=1)

    dfs[direc] = df_mg
    os.chdir("..")
    print()
        
sns.scatterplot(x="comp-1", y="comp-2", hue=df_mg.num_name.tolist(),
                palette=sns.color_palette("hls", 32),
                data=df_mg).set(title="Separating two embs with T-SNE")
        
##sns.move_legend(ax, "center right")
##plt.show()
##fig.savefig('samplefigure', bbox_inches='tight')
        
