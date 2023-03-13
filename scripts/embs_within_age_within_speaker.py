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

scripts_dir = os.getcwd()
os.chdir("../data")
data_dir = os.getcwd()
os.chdir("../results")
results_dir = os.getcwd()
os.chdir("../metadata")
meta_dir = os.getcwd()
os.chdir(scripts_dir)

def get_cossim_df(num_pairs, parquet_path, save_path):
    print(f"Processing {save_path}")
    #
    NUM_PAIRS = num_pairs
    df = pd.read_parquet(os.path.join(meta_dir, parquet_path))
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
            plt.hist(df[f"score_mean_{speech_length}"].tolist())
            plt.title(f'{(" ").join(save_path.split("_"))} cosine similarity scores for {speech_length} speech length')
            plt.savefig(os.path.join(results_dir, f"{save_path}_{speech_length}_cossim_score.png"))
            plt.close()
        #
        lesbian_flag = [(213, 45, 0), (239, 118, 39), (255, 154, 86), (230, 230, 230), (209, 98, 164), (181, 86, 144), (163, 2, 98)]
        for i, speech_length in enumerate(speech_lengths):
            plt.hist(df[f"score_mean_{speech_length}"].tolist(), label=f"{speech_length}",
            bins=np.arange(0, 1, 0.02), color=tuple(map(lambda x: x/255, lesbian_flag[i]))),
        plt.title(f'{(" ").join(save_path.split("_"))} cosine similarity scores')
        plt.legend()
        plt.savefig(os.path.join(results_dir, f"{save_path}_all_cossim_score.png"))
        plt.close()
        #
    print()
    return df


def get_best_threshold_n_roc(df_within, df_across, NUM_PAIRS=3):
    """get metrics when comparing (across speakers) vs (within speakers at the same age)"""
    scores = dict()
    for speech_length in speech_lengths:
        within_scores = list(reduce(lambda x, y: x + y, [df_within[f"score_{i}_{speech_length}"].tolist() for i in range(1, NUM_PAIRS+1)]))
        across_scores = df_across[f"score_1_{speech_length}"].tolist()
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
            plt.savefig(os.path.join(results_dir, f"within_speaker_within_age_VS_across_speaker_{speech_length}_AOC_CURVE.png"))
            plt.close()
            #
    return scores


def vary_threshold_graph(df_within, df_across, NUM_PAIRS=3):
    """get metrics when comparing (across speakers) vs (within speakers at the same age)"""
    scores = dict()
    for speech_length in speech_lengths:
        within_scores = list(reduce(lambda x, y: x + y, [df_within[f"score_{i}_{speech_length}"].tolist() for i in range(1, NUM_PAIRS+1)]))
        within_max = max(within_scores)
        across_scores = df_across[f"score_1_{speech_length}"].tolist()
        across_min = min(across_scores)
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
        # append data points to list
            # the list should start with the first threshold that yields no false negatives
            # if false_negs == 0:
            if threshold <= across_min:
                threshold_scores = [data]
            # the list shold end with the last threshold that yields no false positives (I think?)
            # elif false_poss == 0:
            elif threshold > within_max:
                threshold_scores.append(data)
                break
            else:
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
            fig.savefig(os.path.join(results_dir, f'within_age_within_age_VS_across_speaker_{speech_length}_acc_vs_fnr_n_fpr.png'),
                        format='png',
                        dpi=100,
                        bbox_inches='tight')
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
    if "age" in df_within.columns:
        df_within["bucket"] = df_within["age"].apply(lambda x: (x-min_age)//bucket_size)
    else:
        df_within["bucket"] = df_within["age1"].apply(lambda x: (x-min_age)//bucket_size)
    df_across["bucket"] = df_across["age1"].apply(lambda x: (x-min_age)//bucket_size)
    num_buckets = df_within.bucket.max()
    for speech_length in speech_lengths:
        sub_scores = dict()
        threshold = thresholds[f"{speech_length}"][0]
        for bucket in range(num_buckets):
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


speech_lengths = ["full", 60, 30, 10, 5, 3, 1]
NUM_PAIRS = 3
GENERATE_GRAPHS = False

across_age_df = get_cossim_df(num_pairs=3,
                              parquet_path="within_speaker_across_age_comparisons.parquet",
                              save_path="within_speaker_across_age")

within_age_df = get_cossim_df(num_pairs=3,
                              parquet_path="within_speaker_within_age_comparisons.parquet",
                              save_path="within_speaker_within_age")

across_speaker_df = get_cossim_df(num_pairs=1,
                                  parquet_path="across_speaker_comparisons.parquet",
                                  save_path="across_speaker")

if GENERATE_GRAPHS:
    for speech_length in speech_lengths:
        plt.hist(list(reduce(lambda x, y: x + y, [across_age_df[f"score_{i}_{speech_length}"].tolist() for i in range(1, NUM_PAIRS)])),
                label="within speakers across ages")
        plt.hist(across_speaker_df[f"score_1_{speech_length}"].tolist(),
                label="across speakers")
        plt.legend()
        plt.title(f'across speaker VS within speaker across ages cosine similarity scores for {speech_length} speech length')
        plt.savefig(os.path.join(results_dir, f"within_speaker_across_age_VS_across_speaker_{speech_length}_cossim_score.png"))
        plt.close()
#
    for speech_length in speech_lengths:
        plt.hist(list(reduce(lambda x, y: x + y, [within_age_df[f"score_{i}_{speech_length}"].tolist() for i in range(1, NUM_PAIRS)])),
                label="within speakers within ages")
        plt.hist(across_speaker_df[f"score_1_{speech_length}"].tolist(),
                label="across speakers")
        plt.legend()
        plt.title(f'across speaker VS within speaker within ages cosine similarity scores for {speech_length} speech length')
        plt.savefig(os.path.join(results_dir, f"within_speaker_within_age_VS_across_speaker_{speech_length}_cossim_score.png"))
        plt.close()

scores = get_best_threshold_n_roc(within_age_df, across_speaker_df, 3)
thresh_scores = vary_threshold_graph(within_age_df, across_speaker_df, 3)
add_threshold_accs_to_df(within_age_df, across_age_df, across_speaker_df, scores, 3)
overall_accuracies = overall_accuracy(across_age_df, across_speaker_df, scores, 3)
across_ages_accuracies = across_ages_accuracy(across_age_df, across_speaker_df, scores, 3)
within_ages_bucket_accuracies = per_bucket_accuracy(within_age_df, across_speaker_df, thresholds, num_pairs=3)
across_ages_bucket_accuracies = per_bucket_accuracy(across_age_df, across_speaker_df, thresholds, num_pairs=3)

# print overall_accuracies
if True:
    print(f"{'length':>6}\t{'threshold':>9}\t{'accuracy':>8}\t{'false_negs':>10}\t{'false_poss':>10}\t{'fnr':>5}\t{'fpr':>5}")
    for key, value in overall_accuracies.items(): 
        print(f"{key:>6}\t{value[0]*100:>9.2f}\t{value[1]*100:>8.2f}\t{value[2]:>10}\t{value[3]:>10}\t{value[4]*100:>5.2f}\t{value[5]*100:>5.2f}")
    print()

# print across_ages_accuracies
if True:
    print("-"*(6+9+9+8+10+10+5+5+(7*7)))
    for key, value in across_ages_accuracies.items():
        print(f"{'length':>6}\t{'age range':>9}\t{'threshold':>9}\t{'accuracy':>8}\t{'false_negs':>10}\t{'false_poss':>10}\t{'fnr':>5}\t{'fpr':>5}")
        for age_key, age_value in value.items():
            print(f"{key:>6}\t{age_key:>9}\t{age_value[0]*100:>9.2f}\t{age_value[1]*100:>8.2f}\t{age_value[2]:>10}\t{age_value[3]:>10}\t{age_value[4]*100:>5.2f}\t{age_value[5]*100:>5.2f}")
        print("-"*(6+9+9+8+10+10+5+5+(7*7)))

# print within_ages_bucket_accuracies
if True:
    print("-"*(6+6+9+8+10+10+5+5+(7*7)))
    for key, value in within_ages_bucket_accuracies.items():
        print(f"{'length':>6}\t{'bucket':>6}\t{'threshold':>9}\t{'accuracy':>8}\t{'false_negs':>10}\t{'false_poss':>10}\t{'fnr':>5}\t{'fpr':>5}")
        for bucket_key, bucket_value in value.items():
            print(f"{key:>6}\t{bucket_key:>6}\t{bucket_value[0]*100:>9.2f}\t{bucket_value[1]*100:>8.2f}\t{bucket_value[2]:>10}\t{bucket_value[3]:>10}\t{bucket_value[4]*100:>5.2f}\t{bucket_value[5]*100:>5.2f}")
        print("-"*(6+6+9+8+10+10+5+5+(7*7)))

# print across_ages_bucket_accuracies
if True:
    print("-"*(6+6+9+8+10+10+5+5+(7*7)))
    for key, value in across_ages_bucket_accuracies.items():
        print(f"{'length':>6}\t{'bucket':>6}\t{'threshold':>9}\t{'accuracy':>8}\t{'false_negs':>10}\t{'false_poss':>10}\t{'fnr':>5}\t{'fpr':>5}")
        for bucket_key, bucket_value in value.items():
            print(f"{key:>6}\t{bucket_key:>6}\t{bucket_value[0]*100:>9.2f}\t{bucket_value[1]*100:>8.2f}\t{bucket_value[2]:>10}\t{bucket_value[3]:>10}\t{bucket_value[4]*100:>5.2f}\t{bucket_value[5]*100:>5.2f}")
        print("-"*(6+6+9+8+10+10+5+5+(7*7)))