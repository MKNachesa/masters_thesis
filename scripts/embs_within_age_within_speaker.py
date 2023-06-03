import pandas as pd
import os
from collections import defaultdict
from random import choice, sample
import pickle as pkl
from sklearn.metrics.pairwise import cosine_similarity
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy import stats
from matplotlib.patches import Rectangle
import seaborn as sns


def get_cossim_df(num_pairs, parquet_path, save_path):
    print(f"Processing {save_path}")
    #
    NUM_PAIRS = num_pairs
    df = pd.read_parquet(os.path.join(meta_dir, parquet_path))
    #
    if "age" in df.columns:
        df["gender"] = df.intressent_id.apply(lambda x: iid_to_gen[x])
    else:
        df["gender1"] = df.intressent_id1.apply(lambda x: iid_to_gen[x])
        df["gender2"] = df.intressent_id2.apply(lambda x: iid_to_gen[x])
    #
    danfs_to_use = set()
    for i in range(1, NUM_PAIRS+1):
        danfs_to_use.update(set(reduce(lambda x, y: list(x) + list(y), list(map(set, df[f"pair_{i}"].tolist())))))
    #
    dokids_to_use = set(i.split("_")[0] for i in danfs_to_use)
    #
    for speech_length in speech_lengths:
        print(f"Processing {speech_length}")
        length_dir = os.path.join(data_dir, str(speech_length))
        embs = dict()
        for emb_path in next(os.walk(length_dir))[2]:
            if emb_path.split("_")[1].split(".")[0] in dokids_to_use:
                with open(os.path.join(length_dir, emb_path), "rb") as infile:
                    dokid_embs = pkl.load(infile)
                for danf, emb in dokid_embs.items():
                    if danf in danfs_to_use:
                        embs[danf] = emb
        for i in range(1, NUM_PAIRS+1):
            df[f"score_{i}_{speech_length}"] = df[f"pair_{i}"].apply(lambda y: cosine_similarity(*list(map(lambda x: embs[x], y)))[0][0])
        df[f"score_mean_{speech_length}"] = df[[f"score_{i}_{speech_length}" for i in range(1, NUM_PAIRS+1)]].apply(lambda x: sum(x)/NUM_PAIRS, axis=1)
        #
    if GENERATE_GRAPHS:
        for speech_length in speech_lengths:
            plt.hist(list(reduce(lambda x, y: x + y, [df[f"score_{i}_{speech_length}"].tolist() for i in range(1, NUM_PAIRS+1)])))
            plt.title(f'{(" ").join(save_path.split("_"))} cosine similarity scores for {speech_length} speech length')
            plt.savefig(os.path.join(results_dir, "unneeded", f"{save_path}_{speech_length}_cossim_score.png"))
            plt.close()
        #
        #lesbian_flag = [(213, 45, 0), (239, 118, 39), (255, 154, 86), (230, 230, 230), (209, 98, 164), (181, 86, 144), (163, 2, 98)]
        for i, speech_length in enumerate(speech_lengths):
            plt.hist(list(reduce(lambda x, y: x + y, [df[f"score_{i}_{speech_length}"].tolist() for i in range(1, NUM_PAIRS+1)])),
                     label=f"{speech_length}",
                     bins=np.arange(-0.2, 1, 0.02))#,
                     #alpha=0.5)#, color=tuple(map(lambda x: x/255, lesbian_flag[i]))),
        plt.title(f'{(" ").join(save_path.split("_"))} cosine similarity scores')
        plt.legend()
        plt.savefig(os.path.join(results_dir, "overal average cossim scores", f"{save_path}_all_cossim_score.png"))
        plt.close()
        #
    print()
    return df


def get_best_threshold_n_roc(df_within, df_across, split="train", NUM_PAIRS=3, mode="across"):
    """get metrics when comparing (across speakers) vs (within speakers at the same age)"""
    if mode == "across":
        mode = ""
    else:
        mode = "_within"
    scores = dict()
    within_scores_all = []
    across_scores_all = []
    for speech_length in speech_lengths:
        within_scores = list(reduce(lambda x, y: x + y, [df_within[f"score_{i}_{speech_length}"].tolist() for i in range(1, NUM_PAIRS+1)]))
        across_scores = df_across[f"score_1_{speech_length}"].tolist()
        within_scores_all += within_scores
        across_scores_all += across_scores
        across_n_within_age_scores = across_scores + within_scores
        true_scores = [0 for _ in range(len(across_scores))] + [1 for _ in range(len(within_scores))]
        fpr, tpr, thresholds = roc_curve(true_scores, across_n_within_age_scores, pos_label=1)
        roc_auc = auc(fpr, tpr)
        #
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        threshold = interp1d(fpr, thresholds)(eer)
        threshold = float(threshold)
        #
        scores[f"{speech_length}"] = (threshold, (fpr, tpr, thresholds))
        #
        if GENERATE_GRAPHS:
            plt.title(f'ROC curve for {speech_length} speech length')
            plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
            plt.legend(loc = 'lower right')
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.savefig(os.path.join(results_dir, "roc curves", f"{split}_AOC_CURVE_within_speaker_within_age_VS_across_speaker{mode}_{speech_length}.png"))
            plt.close()
            #
    across_n_within_age_scores = across_scores_all + within_scores_all
    true_scores = [0 for _ in range(len(across_scores_all))] + [1 for _ in range(len(within_scores_all))]
    fpr, tpr, thresholds = roc_curve(true_scores, across_n_within_age_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    threshold = interp1d(fpr, thresholds)(eer)
    threshold = float(threshold)
    scores["all"] = (threshold, (fpr, tpr, thresholds))
    if GENERATE_GRAPHS:
        plt.title(f'ROC curve for all speech lengths')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(os.path.join(results_dir, "roc curves", f"{split}_AOC_CURVE_within_speaker_within_age_VS_across_speaker{mode}_all_speech_lengths.png"))
        plt.close()
    return scores


def vary_threshold_graph(df_within, df_across, split="train", NUM_PAIRS=3, mode="across"):
    # """get metrics when comparing (across speakers) vs (within speakers at the same age)"""
    if mode == "across":
        mode = ""
    else:
        mode = "_within"
    scores = dict()
    for speech_length in speech_lengths:
        within_scores = list(reduce(lambda x, y: x + y, [df_within[f"score_{i}_{speech_length}"].tolist() for i in range(1, NUM_PAIRS+1)]))
        # within_max = max(within_scores)
        across_scores = df_across[f"score_1_{speech_length}"].tolist()
        # across_min = min(across_scores)
        #
        threshold_scores = []
        for threshold in np.arange(-0.2, 1.01, 0.01):
            thresh_across_scores = [1 if i >= threshold else 0 for i in across_scores]
            thresh_within_scores = [1 if i >= threshold else 0 for i in within_scores]
            across_n_within_age_scores = thresh_across_scores + thresh_within_scores
            true_scores = [0 for _ in range(len(thresh_across_scores))] + [1 for _ in range(len(thresh_within_scores))]
            accuracy = sum([1 if across_n_within_age_scores[i] == true_scores[i] else 0 for i in range(len(true_scores))])/len(true_scores)
            false_negs = sum([1 if (across_n_within_age_scores[i] != true_scores[i]) and (true_scores[i] == 1) else 0
                              for i in range(len(true_scores))])
            fnr = false_negs/len(thresh_within_scores)
            false_poss = sum([1 if (across_n_within_age_scores[i] != true_scores[i]) and (true_scores[i] == 0) else 0
                              for i in range(len(true_scores))])
            fpr = false_poss/len(thresh_across_scores)
            data = (threshold, accuracy, false_negs, false_poss, fnr, fpr)
            threshold_scores.append(data)
            #
        scores[f"{speech_length}"] = threshold_scores
        #
        if GENERATE_GRAPHS:
            fig, ax = plt.subplots()
            thresholds = [i[0] for i in threshold_scores]
            accs = [i[1] for i in threshold_scores]
            fnrs = [i[4] for i in threshold_scores]
            fprs = [i[5] for i in threshold_scores]
            #
            ax.plot(thresholds, accs, color="red", label="accuracy")
            ax.set_xlabel("threshold", fontsize = 14)
            ax.set_ylabel("accuracy",
                        color="red",
                        fontsize=14)
            ax.set_ylim(top=1.024396551724138)
            ax.spines['left'].set_color('red')
            ax.yaxis.label.set_color('red')
            ax.tick_params(axis='y', colors='red')
            ax.set_ylim(top=1.024396551724138)
            plt.legend(loc="upper right")
            #
            ax2 = ax.twinx()
            ax2.plot(thresholds, fnrs, label="fnr")
            ax2.set_ylabel("fnr/fpr",color="black",fontsize=14)
            #
            ax2.plot(thresholds, fprs, color="orange", label="fpr")
            ax2.set_ylim(top=1.024396551724138)
            plt.legend()
            plt.title(f"Accuracy VS FNR/FNR at different thresholds for {speech_length} speech length")
            fig.savefig(os.path.join(results_dir, "acc vs fnr and fpr graphs", f'{split}_acc_vs_fnr_n_fpr_within_age_within_age_VS_across_speaker{mode}_{speech_length}.png'),
                        format='png',
                        dpi=100,
                        bbox_inches='tight')
            plt.close()
            #
    return scores


def add_threshold_accs_to_df(within_age_df, across_age_df, across_speaker_df, thresholds, num_pairs=3):
    for speech_length in speech_lengths:
        threshold = thresholds[f"{speech_length}"][0]
        for i in range(1, num_pairs+1):
            across_age_df[f"thresh_score_{i}_{speech_length}"] = across_age_df[f"score_{i}_{speech_length}"].apply(
                lambda x: 1 if x >= threshold else 0)
        for i in range(1, num_pairs+1):
            within_age_df[f"thresh_score_{i}_{speech_length}"] = within_age_df[f"score_{i}_{speech_length}"].apply(
                lambda x: 1 if x >= threshold else 0)
        across_speaker_df[f"thresh_score_1_{speech_length}"] = across_speaker_df[f"score_1_{speech_length}"].apply(
            lambda x: 1 if x >= threshold else 0)


def overall_accuracy(across_age_df, across_speaker_df, thresholds, num_pairs=3):
    scores = dict()
    for speech_length in speech_lengths:
        threshold = thresholds[f"{speech_length}"][0]
        #
        across_age_preds = list(reduce(lambda x, y: x + y, 
                                        [across_age_df[f"thresh_score_{i}_{speech_length}"].tolist() for i in range(1, num_pairs+1)]))
        across_speaker_preds = across_speaker_df[f"thresh_score_1_{speech_length}"].tolist()
        #
        all_preds = across_age_preds + across_speaker_preds
        all_true = [1 for _ in range(len(across_age_preds))] + [0 for _ in range(len(across_speaker_preds))]
        #
        accuracy = sum([1 if all_preds[i] == all_true[i] else 0 for i in range(len(all_true))])/len(all_true)
        #
        false_negs = sum([1 if (all_preds[i] != all_true[i]) and (all_true[i] == 1) else 0
                            for i in range(len(all_true))])
        fnr = false_negs/len(across_age_preds)
        #
        false_poss = sum([1 if (all_preds[i] != all_true[i]) and (all_true[i] == 0) else 0
                            for i in range(len(all_true))])
        fpr = false_poss/len(across_speaker_preds)
        #
        data = (threshold, accuracy, false_negs, false_poss, fnr, fpr)
        scores[f"{speech_length}"] = data
    return scores


def across_ages_accuracy(across_age_df, across_speaker_df, thresholds, num_pairs=3):
    scores = dict()
    across_age_df["age_diff"] = across_age_df[["age1", "age2"]].apply(lambda x: x.age2-x.age1, axis=1)
    across_speaker_df["age_diff"] = across_speaker_df[["age1", "age2"]].apply(lambda x: x.age2-x.age1, axis=1)
    for speech_length in speech_lengths:
        sub_scores = dict()
        threshold = thresholds[f"{speech_length}"][0]
        for age_diff in range(0, 9+1):
            across_age_preds = list(reduce(lambda x, y: x + y, 
                                            [across_age_df[across_age_df["age_diff"] == age_diff]
                                            [f"thresh_score_{i}_{speech_length}"].tolist() for i in range(1, num_pairs+1)]))
            across_speaker_preds = across_speaker_df[across_speaker_df["age_diff"] == age_diff][f"thresh_score_1_{speech_length}"].tolist()
            #
            all_preds = across_age_preds + across_speaker_preds
            all_true = [1 for _ in range(len(across_age_preds))] + [0 for _ in range(len(across_speaker_preds))]
            #
            accuracy = sum([1 if all_preds[i] == all_true[i] else 0 for i in range(len(all_true))])/len(all_true)
            #
            false_negs = sum([1 if (all_preds[i] != all_true[i]) and (all_true[i] == 1) else 0
                                for i in range(len(all_true))])
            fnr = false_negs/len(across_age_preds)
            #
            false_poss = sum([1 if (all_preds[i] != all_true[i]) and (all_true[i] == 0) else 0
                                for i in range(len(all_true))])
            fpr = false_poss/len(across_speaker_preds)
            #
            data = (threshold, accuracy, false_negs, false_poss, fnr, fpr)
            sub_scores[f"{age_diff}"] = data
        scores[f"{speech_length}"] = sub_scores
    return scores


def per_bucket_accuracy(df_within, df_across, thresholds, num_pairs=3, bucket_size=5):
    scores = dict()
    min_age = df_across.age1.min()
    bucket_calc = lambda x: f"{(x-min_age)//bucket_size*5+min_age}-{(x-min_age)//bucket_size*5+min_age+4}"
    if "age" in df_within.columns:
        df_within["bucket"] = df_within["age"].apply(bucket_calc)
    else:
        df_within["bucket"] = df_within["age1"].apply(bucket_calc)
    df_across["bucket"] = df_across["age1"].apply(bucket_calc)
    num_buckets = df_within.bucket.max()
    buckets = sorted(list(set(df_within.bucket.tolist())))
    for speech_length in speech_lengths:
        sub_scores = dict()
        threshold = thresholds[f"{speech_length}"][0]
        for bucket in buckets:
            within_speaker_preds = list(reduce(lambda x, y: x + y, 
                                            [df_within[df_within["bucket"] == bucket][f"thresh_score_{i}_{speech_length}"].tolist() for i in range(1, num_pairs+1)]))
            across_speaker_preds = df_across[df_across["bucket"] == bucket][f"thresh_score_1_{speech_length}"].tolist()
            #
            all_preds = within_speaker_preds + across_speaker_preds
            all_true = [1 for _ in range(len(within_speaker_preds))] + [0 for _ in range(len(across_speaker_preds))]
            #
            accuracy = sum([1 if all_preds[i] == all_true[i] else 0 for i in range(len(all_true))])/len(all_true)
            #
            false_negs = sum([1 if (all_preds[i] != all_true[i]) and (all_true[i] == 1) else 0
                                for i in range(len(all_true))])
            fnr = false_negs/len(within_speaker_preds)
            #
            false_poss = sum([1 if (all_preds[i] != all_true[i]) and (all_true[i] == 0) else 0
                                for i in range(len(all_true))])
            fpr = false_poss/len(across_speaker_preds)
            #
            data = (threshold, accuracy, false_negs, false_poss, fnr, fpr)
            sub_scores[f"{bucket}"] = data
        scores[f"{speech_length}"] = sub_scores
    return scores


def get_cossim_dfs(split):
    across_age_df = get_cossim_df(num_pairs=3,
                                parquet_path=f"{split}_within_speaker_across_age_comparisons.parquet",
                                save_path=f"{split}_within_speaker_across_age")
    #
    within_age_df = get_cossim_df(num_pairs=3,
                                parquet_path=f"{split}_within_speaker_within_age_comparisons.parquet",
                                save_path=f"{split}_within_speaker_within_age")
    #
    across_speaker_df = get_cossim_df(num_pairs=1,
                                    parquet_path=f"{split}_across_speaker_comparisons.parquet",
                                    save_path=f"{split}_across_speaker")
    #
    across_speaker_within_age_df = get_cossim_df(num_pairs=1,
                                                parquet_path=f"{split}_across_speaker_within_age_comparisons.parquet",
                                                save_path=f"{split}_across_speaker_within_age")
    #
    cossims = {"across speaker across age": [],
               "within speaker across age": [],
               "within speaker within age": []}
    for i, speech_length in enumerate(speech_lengths):
        for key, df, num_pairs in [("across speaker across age", across_speaker_df, 1),
                                   ("within speaker across age", across_age_df, 3),
                                   ("within speaker within age", within_age_df, 3)]:
            
            cossims[key] += list(reduce(lambda x, y: x + y, [df[f"score_{i}_{speech_length}"].tolist() for i in range(1, num_pairs+1)]))
    for df_type, cossims_list in cossims.items():
        plt.hist(cossims_list, label=df_type, bins=np.arange(-0.2, 1, 0.02))
    plt.title(f'cosine similarity scores for all speech lengths combined')
    plt.legend()
    plt.savefig(os.path.join(results_dir, "overal average cossim scores", f"{split}_all_speech_lengths_all_cossim_score.png"))
    plt.close()
    #
    return across_age_df, within_age_df, across_speaker_df, across_speaker_within_age_df


scripts_dir = os.getcwd()
os.chdir("../data")
data_dir = os.getcwd()
os.chdir("../results")
results_dir = os.getcwd()
os.chdir("../metadata")
meta_dir = os.getcwd()
os.chdir(scripts_dir)


speech_lengths = ["full", 60, 30, 10, 5, 3, 1]
NUM_PAIRS = 3
GENERATE_GRAPHS = True
speaker_meta_path = os.path.join(meta_dir, "person.csv")


meta_df = pd.read_csv(speaker_meta_path)
iid_to_gen = dict(map(lambda x: (x[0], "F") if x[1] == "kvinna" else (x[0], "M"), meta_df[["Id", "Kön"]].itertuples(index=False, name=None)))
train_across_age_df, train_within_age_df, train_across_speaker_df, train_across_speaker_within_age_df = get_cossim_dfs("train")
dev_across_age_df, dev_within_age_df, dev_across_speaker_df, dev_across_speaker_within_age_df = get_cossim_dfs("dev")
test_across_age_df, test_within_age_df, test_across_speaker_df, test_across_speaker_within_age_df = get_cossim_dfs("test")


def within_vs_across_speaker_graphs(across_age_df, within_age_df, across_speaker_df, split="train", mode="across"):
    for speech_length in speech_lengths:
        plt.hist(list(reduce(lambda x, y: x + y, [across_age_df[f"score_{i}_{speech_length}"].tolist() for i in range(1, NUM_PAIRS+1)])),
                label="within speakers across ages", alpha=0.5)#, histtype="step")
        plt.hist(across_speaker_df[f"score_1_{speech_length}"].tolist(),
                label=f"across speakers {mode} ages", alpha=0.5)#, histtype="step")
        plt.legend()
        plt.title(f'{split} across speaker VS within speaker across ages cosine similarity scores for {speech_length} speech length')
        plt.savefig(os.path.join(results_dir, 
                                 "within speaker vs across speaker graphs", 
                                 f"{split}_within_speaker_across_age_VS_across_speaker_{speech_length}_cossim_score.png"))
        plt.close()
        #
    for speech_length in speech_lengths:
        plt.hist(list(reduce(lambda x, y: x + y, [within_age_df[f"score_{i}_{speech_length}"].tolist() for i in range(1, NUM_PAIRS+1)])),
                label="within speakers within ages", alpha=0.5)#, histtype="step")
        plt.hist(across_speaker_df[f"score_1_{speech_length}"].tolist(),
                label=f"across speakers {mode} ages", alpha=0.5)#, histtype="step")
        plt.legend()
        plt.title(f'{split} across speaker {mode} ages VS within speaker within ages\ncosine similarity scores for {speech_length} speech length')
        plt.savefig(os.path.join(results_dir, 
                                 "within speaker vs across speaker graphs",
                                 f"{split}_within_speaker_within_age_VS_across_speaker_{mode}_age_{speech_length}_cossim_score.png"))
        plt.close()


within_vs_across_speaker_graphs(train_across_age_df, train_within_age_df, train_across_speaker_df, split="train", mode="across")
within_vs_across_speaker_graphs(train_across_age_df, train_within_age_df, train_across_speaker_within_age_df, split="train", mode="within")


def across_speaker_within_vs_across_age_graphs(across_age_df, within_age_df, split="train"):
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    stats_df = pd.DataFrame(columns=["comparison", "length", "mean_vp", "std_vp", "ci_lower", "ci_upper"])
    for speech_length in speech_lengths:
        across_age = across_age_df[f"score_1_{speech_length}"].tolist()
        within_age = within_age_df[f"score_1_{speech_length}"].tolist()
        plt.hist(across_age,
                label=f"across age", alpha=0.5, bins=np.arange(-0.2, 1, 0.02))#, histtype="step")
        plt.hist(within_age,
                label=f"within age", alpha=0.5, bins=np.arange(-0.2, 1, 0.02))#, histtype="step")
        #
        mean = np.mean(across_age)
        std = np.std(across_age)
        t = np.abs(stats.t.ppf((1-0.95)/2, len(across_age)-1))
        ci = (mean-std*t/np.sqrt(len(across_age)), mean+std*t/np.sqrt(len(across_age)))
        stats_dict = {"comparison": "across", "length": speech_length, "mean_vp": np.mean(across_age),
                      "std_vp": np.std(across_age), "ci_lower":ci[0], "ci_upper":ci[1]}
        stats_df = pd.concat([pd.DataFrame(stats_dict, index=[0]), stats_df]).reset_index(drop=True)
        mean = np.mean(within_age)
        std = np.std(within_age)
        t = np.abs(stats.t.ppf((1-0.95)/2, len(within_age)-1))
        ci = (mean-std*t/np.sqrt(len(within_age)), mean+std*t/np.sqrt(len(within_age)))
        stats_dict = {"comparison": "within", "length": speech_length, "mean_vp": np.mean(within_age),
                      "std_vp": np.std(within_age), "ci_lower":ci[0], "ci_upper":ci[1]}
        stats_df = pd.concat([pd.DataFrame(stats_dict, index=[0]), stats_df]).reset_index(drop=True)
        #
        stat, pvalue = stats.ttest_rel(across_age, within_age)
        text = f"p = {pvalue:.2e}"
        y_lim = plt.gca().get_ylim()[1]*0.95
        plt.text(0.05, y_lim, text, verticalalignment="top", bbox=props)
        plt.legend(title="Comparison type")
        plt.title(f'{split} across speaker within VS across age cosine similarity scores for {speech_length} speech length')
        plt.savefig(os.path.join(results_dir, 
                                 "within speaker graphs",
                                 f"{split}_across_speaker_across_VS_within_age_{speech_length}_cossim_score.png"))
        plt.close()
    df = stats_df[stats_df.comparison=="within"]
    x = [f"{i}" for i in df["length"].tolist()]
    plt.plot(x, df["mean_vp"], label="same")
    plt.gca().fill_between(x, df["ci_lower"], df["ci_upper"], alpha=0.15)
    df = stats_df[stats_df.comparison=="across"]
    x = [f"{i}" for i in df["length"].tolist()]
    plt.plot(x, df["mean_vp"], label="different")
    plt.gca().fill_between(x, df["ci_lower"], df["ci_upper"], alpha=0.15)
    plt.title(f'{split} cosine similarity score means for all speech lengths,\ndifferent speakers, same VS different age')
    plt.legend(title="age")
    plt.savefig(os.path.join(results_dir, 
                             "within speaker graphs",
                             f"{split}_across_speaker_same_VS_diff_age_means_cossim_score.png"))
    plt.close()


across_speaker_within_vs_across_age_graphs(train_across_speaker_df, train_across_speaker_within_age_df, split="train")


def compare_men_women_within_age(within_age_df, split, mode="within"):
    print(f"speech length\tstatistic\tp-value\n----------------------------------------")
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    for speech_length in speech_lengths:
        men = list(reduce(lambda x, y: x + y, [within_age_df[within_age_df.gender=="M"][
            f"score_{i}_{speech_length}"].tolist() for i in range(1, NUM_PAIRS+1)]))
        women = list(reduce(lambda x, y: x + y, [within_age_df[within_age_df.gender=="F"][
            f"score_{i}_{speech_length}"].tolist() for i in range(1, NUM_PAIRS+1)]))
        #
        stat, pvalue = stats.ttest_ind(men, women)
        text = f"p = {pvalue:.2e}"
        plt.hist(men,
                label="M", alpha=0.5, bins=np.arange(-0.2, 1, 0.02))#, histtype="step")
        plt.hist(women,
                label="F", alpha=0.5, bins=np.arange(-0.2, 1, 0.02))#, histtype="step")
        plt.legend(loc="upper left")
        y_lim = plt.gca().get_ylim()[1]*0.95
        plt.text(0.05, y_lim, text, verticalalignment="top", bbox=props)
        plt.title(f'{split} within {mode} within ages, m VS f cosine similarity scores for {speech_length} speech length')
        plt.savefig(os.path.join(results_dir, "gender graphs", f"{split}_{mode}_speaker_within_age_m_VS_f_{speech_length}_cossim_score.png"))
        plt.close()
        print(f"{speech_length:>13}\t{stat:>9.2f}\t{pvalue:>7.2e}")


compare_men_women_within_age(train_within_age_df, "train")


def compare_men_women_across_speaker(across_speaker_within_age_df, split):
    print(f"speech length\tstatistic\tp-value\n----------------------------------------")
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    NUM_PAIRS=1
    stats_df = pd.DataFrame(columns=["comparison", "length", "mean_vp", "std_vp", "ci_lower", "ci_upper"])
    for speech_length in speech_lengths:
        # men = list(reduce(lambda x, y: x + y, [across_speaker_within_age_df[
        #     (across_speaker_within_age_df.gender1=="M") & (across_speaker_within_age_df.gender2=="M")][
        #     f"score_{i}_{speech_length}"].tolist() for i in range(1, NUM_PAIRS+1)]))
        # women = list(reduce(lambda x, y: x + y, [across_speaker_within_age_df[
        #     (across_speaker_within_age_df.gender1=="F") & (across_speaker_within_age_df.gender2=="F")][
        #     f"score_{i}_{speech_length}"].tolist() for i in range(1, NUM_PAIRS+1)]))
        same_gender = list(reduce(lambda x, y: x + y, [across_speaker_within_age_df[
            (across_speaker_within_age_df.gender1==across_speaker_within_age_df.gender2)][
            f"score_{i}_{speech_length}"].tolist() for i in range(1, NUM_PAIRS+1)]))
        men_vs_women = list(reduce(lambda x, y: x + y, [across_speaker_within_age_df[
            (across_speaker_within_age_df.gender1!=across_speaker_within_age_df.gender2)][
            f"score_{i}_{speech_length}"].tolist() for i in range(1, NUM_PAIRS+1)]))
        #
        mean = np.mean(same_gender)
        std = np.std(same_gender)
        t = np.abs(stats.t.ppf((1-0.95)/2, len(same_gender)-1))
        ci = (mean-std*t/np.sqrt(len(same_gender)), mean+std*t/np.sqrt(len(same_gender)))
        stats_dict = {"comparison": "same", "length": speech_length, "mean_vp": np.mean(same_gender),
                      "std_vp": np.std(same_gender), "ci_lower":ci[0], "ci_upper":ci[1]}
        stats_df = pd.concat([pd.DataFrame(stats_dict, index=[0]), stats_df]).reset_index(drop=True)
        #
        # print("1", speech_length)
        mean = np.mean(men_vs_women)
        std = np.std(men_vs_women)
        t = np.abs(stats.t.ppf((1-0.95)/2, len(men_vs_women)-1))
        ci = (mean-std*t/np.sqrt(len(men_vs_women)), mean+std*t/np.sqrt(len(men_vs_women)))
        stats_dict = {"comparison": "diff", "length": speech_length, "mean_vp": np.mean(men_vs_women),
                      "std_vp": np.std(men_vs_women), "ci_lower":ci[0], "ci_upper":ci[1]}
        stats_df = pd.concat([pd.DataFrame(stats_dict, index=[0]), stats_df]).reset_index(drop=True)
        # stat, pvalue = stats.f_oneway(men, women, men_vs_women)
        stat, pvalue = stats.ttest_ind(same_gender, men_vs_women)
        text = f"p = {pvalue:.2e}"
        # plt.hist(men,
        #         label="M", alpha=0.5, bins=np.arange(-0.2, 1, 0.02))#, histtype="step")
        # plt.hist(women,
        #         label="F", alpha=0.5, bins=np.arange(-0.2, 1, 0.02))#, histtype="step")
        plt.hist(same_gender,
                label="same gender", alpha=0.5, bins=np.arange(-0.2, 1, 0.02))#, histtype="step")
        plt.hist(men_vs_women,
                label="different gender", alpha=0.5, bins=np.arange(-0.2, 1, 0.02))#, histtype="step")
        plt.legend(loc="upper left")
        y_lim = plt.gca().get_ylim()[1]*0.95
        plt.text(0.70, y_lim, text, verticalalignment="top", bbox=props)
        plt.title(f'{split} across speaker within ages,\nsame VS diff gender cosine similarity scores for {speech_length} speech length')
        plt.savefig(os.path.join(results_dir, 
                                 "gender graphs",
                                 f"{split}_across_speaker_within_age_same_VS_diff_gender_{speech_length}_cossim_score.png"))
        plt.close()
        print(f"{speech_length:>13}\t{stat:>9.2f}\t{pvalue:>7.2e}")
        #
    df = stats_df[stats_df.comparison=="same"]
    x = [f"{i}" for i in df["length"].tolist()]
    plt.plot(x, df["mean_vp"], label="same gender")
    plt.gca().fill_between(x, df["ci_lower"], df["ci_upper"], alpha=0.15)
    df = stats_df[stats_df.comparison=="diff"]
    x = [f"{i}" for i in df["length"].tolist()]
    plt.plot(x, df["mean_vp"], label="different gender")
    plt.gca().fill_between(x, df["ci_lower"], df["ci_upper"], alpha=0.15)
    plt.title(f'{split} cosine similarity score means for all speech lengths,\ndifferent speakers, same ages, same VS diff gender')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 
                             "gender graphs",
                             f"{split}_across_speaker_within_age_same_VS_diff_gender_means_cossim_score.png"))
    plt.close()


compare_men_women_across_speaker(train_across_speaker_within_age_df, "train")


scores = get_best_threshold_n_roc(train_within_age_df, train_across_speaker_df, "train", 3)
thresh_scores = vary_threshold_graph(train_within_age_df, train_across_speaker_df, "train", 3)
add_threshold_accs_to_df(train_within_age_df, train_across_age_df, train_across_speaker_df, scores, 3)
add_threshold_accs_to_df(dev_within_age_df, dev_across_age_df, dev_across_speaker_df, scores, 3)
add_threshold_accs_to_df(test_within_age_df, test_across_age_df, test_across_speaker_df, scores, 3)
train_overall_accuracies = overall_accuracy(train_across_age_df, train_across_speaker_df, scores, 3)
dev_overall_accuracies = overall_accuracy(dev_across_age_df, dev_across_speaker_df, scores, 3)
test_overall_accuracies = overall_accuracy(test_across_age_df, test_across_speaker_df, scores, 3)
test_across_ages_accuracies = across_ages_accuracy(test_across_age_df, test_across_speaker_df, scores, 3)
test_within_ages_bucket_accuracies = per_bucket_accuracy(test_within_age_df, test_across_speaker_df, scores, num_pairs=3)
test_across_ages_bucket_accuracies = per_bucket_accuracy(test_across_age_df, test_across_speaker_df, scores, num_pairs=3)

# print overall_accuracies
def print_overall_accuracies(overall_accuracies):
    print(f"{'length':>6}\t{'threshold':>9}\t{'accuracy':>8}\t{'false_negs':>10}\t{'false_poss':>10}\t{'fnr':>5}\t{'fpr':>5}")
    for key, value in overall_accuracies.items(): 
        print(f"{key:>6}\t{value[0]*100:>9.2f}\t{value[1]*100:>8.2f}\t{value[2]:>10}\t{value[3]:>10}\t{value[4]*100:>5.2f}\t{value[5]*100:>5.2f}")
    print()

print_overall_accuracies(train_overall_accuracies)
print_overall_accuracies(dev_overall_accuracies)
print_overall_accuracies(test_overall_accuracies)


def print_comparison_accuracies(bucket_accuracies, split="train", mode="across", thresh_base="across", comparison="age_range"):
    """comparison: "age_range" or "age_diff" (age difference)
    mode: "across" or "within" (always "across" when "comparison"=="age_diff)"""
    stats_df = pd.DataFrame(columns=[comparison, "length", "accuracy", "threshold"])
    print("-"*(6+9+9+8+10+10+5+5+(7*7)))
    for key, value in bucket_accuracies.items():
        print(f"{'length':>6}\t{comparison:>9}\t{'threshold':>9}\t{'accuracy':>8}\t{'false_negs':>10}\t{'false_poss':>10}\t{'fnr':>5}\t{'fpr':>5}")
        for bucket_key, bucket_value in value.items():
            print(f"{key:>6}\t{bucket_key:>9}\t{bucket_value[0]*100:>9.2f}\t{bucket_value[1]*100:>8.2f}\t{bucket_value[2]:>10}\t{bucket_value[3]:>10}\t{bucket_value[4]*100:>5.2f}\t{bucket_value[5]*100:>5.2f}")
            accuracy = bucket_value[1]
            # std = np.std(same_gender)
            # t = np.abs(stats.t.ppf((1-0.95)/2, len(same_gender)-1))
            # ci = (mean-std*t/np.sqrt(len(same_gender)), mean+std*t/np.sqrt(len(same_gender)))
            stats_dict = {comparison: bucket_key, "length": key, "accuracy": accuracy*100, "threshold": bucket_value[0]*100}
            stats_df = pd.concat([stats_df, pd.DataFrame(stats_dict, index=[0])]).reset_index(drop=True)
        print("-"*(6+9+9+8+10+10+5+5+(7*7)))
        #
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    length_col = ["speech length"] + speech_lengths
    threshold_col = ["threshold"]
    line_col = [extra]
    label_empty = [""]
    # fig, ax = plt.subplots(1, 1)
    for speech_length in speech_lengths:
        df = stats_df[stats_df.length==f"{speech_length}"]
        x = [f"{i}" for i in df[comparison].tolist()]
        # print(df.columns, "\n", df.head())
        threshold = df.threshold.iloc[0]
        threshold_col.append(f"{threshold:.2f}")
        img, = plt.plot(x, df["accuracy"], label=f'{f"{speech_length}, {threshold:.2f}":>20}')
        line_col.append(img)
        # plt.gca().fill_between(x, df["ci_lower"], df["ci_upper"], alpha=0.15)
    plt.title(f'{split} accuracy for all speech lengths,\ndifferent speakers for {thresh_base} thresh base')
    legend_handle = line_col + 3 * len(threshold_col) * [extra]
    legend_labels = np.concatenate([label_empty * len(length_col), label_empty * len(length_col), length_col, threshold_col])
    plt.legend(legend_handle, legend_labels, ncol=4, handletextpad=-2, bbox_to_anchor=(1,1))
    plt.gca().set_xlabel(" ".join(comparison.split("_")))
    plt.gca().set_ylabel("accuracy")
    # sns.move_legend(plt.gca(), "center left", bbox_to_anchor=(1,0.5))
    plt.savefig(os.path.join(results_dir, f"{split}_within_speaker_{mode}_age_{thresh_base}_thresh_base_{comparison}_all_lengths_accuracy_curves.png"), bbox_inches="tight")
    plt.close()

print_comparison_accuracies(test_across_ages_accuracies, "test", "across", "across", "age_diff")
print_comparison_accuracies(test_across_ages_bucket_accuracies, "test", "across", "across")
print_comparison_accuracies(test_within_ages_bucket_accuracies, "test", "within", "across")


def across_ages_graph(across_age_df, across_speaker_df, thresholds, split="train", num_pairs=3):
    scores = dict()
    across_age_df["age_diff"] = across_age_df[["age1", "age2"]].apply(lambda x: x.age2-x.age1, axis=1)
    across_speaker_df["age_diff"] = across_speaker_df[["age1", "age2"]].apply(lambda x: x.age2-x.age1, axis=1)
    stats_df = pd.DataFrame(columns=["length", "age_diff", "mean_vp", "std_vp", "ci_lower", "ci_upper"])
    for speech_length in speech_lengths:
        sub_scores = dict()
        # threshold = thresholds[f"{speech_length}"][0]
        for age_diff in range(0, 9+1):
            across_age_preds = list(reduce(lambda x, y: x + y, 
                                            [across_age_df[across_age_df["age_diff"] == age_diff]
                                            [f"score_{i}_{speech_length}"].tolist() for i in range(1, num_pairs+1)]))
            mean = np.mean(across_age_preds)
            std = np.std(across_age_preds)
            t = np.abs(stats.t.ppf((1-0.95)/2, len(across_age_preds)-1))
            ci = (mean-std*t/np.sqrt(len(across_age_preds)), mean+std*t/np.sqrt(len(across_age_preds)))
            stats_dict = {"speaker": "same", "length": speech_length, "age_diff": age_diff, "mean_vp": np.mean(across_age_preds),
                        "std_vp": np.std(across_age_preds), "ci_lower":ci[0], "ci_upper":ci[1]}
            stats_df = pd.concat([pd.DataFrame(stats_dict, index=[0]), stats_df]).reset_index(drop=True)
            plt.hist(across_age_preds, label=f"{age_diff}, count={len(across_age_preds)}", alpha=0.5, bins=np.arange(-0.2, 1, 0.02))
        across_speaker_preds = list(reduce(lambda x, y: x + y, 
                                    [across_speaker_df#[across_speaker_df["age_diff"] == age_diff]
                                    [f"score_{i}_{speech_length}"].tolist() for i in range(1, 1+1)]))
        mean = np.mean(across_speaker_preds)
        std = np.std(across_speaker_preds)
        t = np.abs(stats.t.ppf((1-0.95)/2, len(across_speaker_preds)-1))
        ci = (mean-std*t/np.sqrt(len(across_speaker_preds)), mean+std*t/np.sqrt(len(across_speaker_preds)))
        stats_dict = {"speaker": "diff", "length": speech_length, "age_diff": age_diff, "mean_vp": np.mean(across_speaker_preds),
                    "std_vp": np.std(across_speaker_preds), "ci_lower":ci[0], "ci_upper":ci[1]}
        stats_df = pd.concat([pd.DataFrame(stats_dict, index=[0]), stats_df]).reset_index(drop=True)
        plt.legend(title="age differences")
        plt.title(f'{split} within speaker cosine similarities for {speech_length} speech length')
        plt.savefig(os.path.join(results_dir, "unneeded", f"{split}_within_speaker_across_age_{speech_length}_cossim_score.png"))
        plt.close()
    speech_length = "full"
    within_age_preds = list(reduce(lambda x, y: x + y, 
                                    [across_age_df[across_age_df["age_diff"] == 0]
                                    [f"score_{i}_{speech_length}"].tolist() for i in range(1, num_pairs+1)]))
    for age_diff in range(1, 9+1):
        across_age_preds = list(reduce(lambda x, y: x + y, 
                                        [across_age_df[across_age_df["age_diff"] == age_diff]
                                        [f"score_{i}_{speech_length}"].tolist() for i in range(1, num_pairs+1)]))
        across_speaker_preds = list(reduce(lambda x, y: x + y, 
                                        [across_speaker_df#[across_speaker_df["age_diff"] == age_diff]
                                        [f"score_{i}_{speech_length}"].tolist() for i in range(1, 1+1)]))
        plt.hist(within_age_preds, label=f"0, within speaker, count={len(within_age_preds)}", alpha=0.5, bins=np.arange(-0.2, 1, 0.02))
        plt.hist(across_age_preds, label=f"{age_diff}, within speaker, count={len(across_age_preds)}", alpha=0.5, bins=np.arange(-0.2, 1, 0.02))
        plt.hist(across_speaker_preds, label=f"all ages, across speaker, count={len(across_speaker_preds)}", alpha=0.5, bins=np.arange(-0.2, 1, 0.02))
        plt.gca().set_ylim(0, 50)
        plt.legend(title="age differences")
        plt.title(f'{split} within and across speaker across ages cosine similarities for {age_diff} year age difference')
        plt.savefig(os.path.join(results_dir, 
                                 "within speaker graphs", 
                                 f"{split}_within_speaker_across_age_{age_diff}_age_diff_cossim_score.png"))
        plt.close()
    for age_diff in range(0, 9+1):
    # for speech_length in speech_lengths:
        df = stats_df[(stats_df.age_diff==age_diff) & (stats_df.speaker=="same")]
        x = [f"{i}" for i in df["length"].tolist()]
        plt.plot(x, df["mean_vp"], label=f"{age_diff}")
        plt.gca().fill_between(x, df["ci_lower"], df["ci_upper"], alpha=0.15)
    df = stats_df[(stats_df.age_diff==age_diff) & (stats_df.speaker=="diff")]
    x = [f"{i}" for i in df["length"].tolist()]
    plt.plot(x, df["mean_vp"], label=f"diff speaker")
    plt.gca().fill_between(x, df["ci_lower"], df["ci_upper"], alpha=0.15)
    plt.title(f'{split} cosine similarity score means for all speech lengths,\nsame VS diff speakers, across ages')
    plt.legend(title="Age differences")
    plt.savefig(os.path.join(results_dir, 
                             "within speaker vs across speaker graphs",
                             f"{split}_within_speaker_across_age_all_speech_lengths_means_cossim_score.png"))
    plt.close()

across_ages_graph(test_across_age_df, test_across_speaker_df, scores, split="test")


scores = get_best_threshold_n_roc(train_within_age_df, train_across_speaker_within_age_df, "train", 3, mode="within")
thresh_scores = vary_threshold_graph(train_within_age_df, train_across_speaker_within_age_df, "train", 3, mode="within")
add_threshold_accs_to_df(train_within_age_df, train_across_age_df, train_across_speaker_within_age_df, scores, 3)
add_threshold_accs_to_df(dev_within_age_df, dev_across_age_df, dev_across_speaker_within_age_df, scores, 3)
add_threshold_accs_to_df(test_within_age_df, test_across_age_df, test_across_speaker_within_age_df, scores, 3)
train_overall_accuracies = overall_accuracy(train_across_age_df, train_across_speaker_within_age_df, scores, 3)
dev_overall_accuracies = overall_accuracy(dev_across_age_df, dev_across_speaker_within_age_df, scores, 3)
test_overall_accuracies = overall_accuracy(test_across_age_df, test_across_speaker_within_age_df, scores, 3)
test_across_ages_accuracies = across_ages_accuracy(test_across_age_df, test_across_speaker_df, scores, 3)
test_within_ages_bucket_accuracies = per_bucket_accuracy(test_within_age_df, test_across_speaker_within_age_df, scores, num_pairs=3)
test_across_ages_bucket_accuracies = per_bucket_accuracy(test_across_age_df, test_across_speaker_within_age_df, scores, num_pairs=3)

print_comparison_accuracies(test_across_ages_accuracies, "test", "across", "within", "age_diff")
print_comparison_accuracies(test_across_ages_bucket_accuracies, "test", "across", "within")
print_comparison_accuracies(test_within_ages_bucket_accuracies, "test", "within", "within")

print_overall_accuracies(train_overall_accuracies)
print_overall_accuracies(dev_overall_accuracies)
print_overall_accuracies(test_overall_accuracies)