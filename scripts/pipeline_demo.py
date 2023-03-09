import pandas as pd
import os
from collections import defaultdict
from random import choice, sample
from sklearn.metrics.pairwise import cosine_similarity

# within speaker within age
NUM_PAIRS = 3
AGE_RANGE = 10

df = pd.read_parquet("../metadata/bucketed_speeches.parquet")

ids_to_dokanf = df.groupby(
    ["intressent_id", "age"]).agg(list)["dokid_anfnummer"].to_dict()

# get starting ages for each speaker
ids_to_min_age = dict()
for iid, age in ids_to_dokanf.keys():
    if age < ids_to_min_age.get(iid, 100):  
            ids_to_min_age[iid] = age

# for each speaker, only keep speakers + debates within range of (starting_age, starting_age + AGE_RANGE-1) (e.g. 20-29)
ids_to_dokanf = dict(filter(lambda x: x[0][1] < ids_to_min_age[x[0][0]] + AGE_RANGE, ids_to_dokanf.items()))

pairs = defaultdict(set)

# first pass
# for each speaker at each age, find pairs of speeches such that they *don't* come from the same debate
for ids, dokid_anfnummers in ids_to_dokanf.items():
    tmp_dokid_anfnummers = dokid_anfnummers[:]
    dokids = set(i.split("_")[0] for i in tmp_dokid_anfnummers)
#
    # find speeches from different debates
    # stop if all remaining speeches are in the same debate or there are NUM_PAIRS pairs
    while len(dokids) > 1 and len(pairs[ids]) < NUM_PAIRS:
        danf1 = choice(tmp_dokid_anfnummers)
        danf2 = choice(tmp_dokid_anfnummers)
        while danf1.split("_")[0] == danf2.split("_")[0]:
            danf2 = choice(tmp_dokid_anfnummers)
#
        del tmp_dokid_anfnummers[tmp_dokid_anfnummers.index(danf1)]
        del tmp_dokid_anfnummers[tmp_dokid_anfnummers.index(danf2)]
        pairs[ids].add(tuple(sorted([danf1, danf2])))
#
        dokids = set([i.split("_")[0] for i in tmp_dokid_anfnummers])

# second pass
# for some speakers it isn't possible to find pairs such that for each both speeches come from a different debate
# fill up speeches pairs for within-speaker comparisons such that all speakers at all ages have NUM_PAIRS comparison pairs
for ids, dokid_anfnummers in ids_to_dokanf.items():
    while len(pairs[ids]) < NUM_PAIRS:
        new_pair = set()
        while len(new_pair) < 2:
            new_pair.add(choice(dokid_anfnummers))
        pairs[ids].add(tuple(sorted(list(new_pair))))


# save the pairs to a dataframe
new_df = pd.DataFrame.from_dict(pairs, columns=[f"pair_{i+1}" for i in range(NUM_PAIRS)], orient="index")
new_df = new_df.reset_index()
new_df[["intressent_id", "age"]] = pd.DataFrame(new_df["index"].tolist(), index=new_df.index)
new_df = new_df.drop("index", axis=1)
new_df = new_df[["intressent_id", "age"] + [f"pair_{i+1}" for i in range(NUM_PAIRS)]]

new_df.to_parquet("../metadata/within_speaker_within_age_comparisons.parquet")


# across speakers
# create speech pairs between different speakers
comparisons = dict()
NUM_PAIRS = 3

def get_pair(dokid_anfnummers1, dokid_anfnummers2):
    """returns a pair of speeches such that, if possible, they do not come from the same debate"""
#
    dokids = set([i.split("_")[0] for i in dokid_anfnummers1] + [i.split("_")[0] for i in dokid_anfnummers2])
    danf1 = choice(dokid_anfnummers1)
    danf2 = choice(dokid_anfnummers2)
    if len(dokids) > 1:
        while (danf1 == danf2) or (danf1.split("_")[0] == danf2.split("_")[0]):
            danf1 = choice(dokid_anfnummers1)
            danf2 = choice(dokid_anfnummers2)
    # else:
    #     while (danf1 == danf2):
    #         danf1 = choice(dokid_anfnummers1)
    #         danf2 = choice(dokid_anfnummers2)
    pair = (danf1, danf2)
    return pair

# for each speaker at each age, pair a speech of theirs with a speech from NUM_PAIRS other speakers
for ids1, dokid_anfnummers1 in ids_to_dokanf.items():
    for _ in range(NUM_PAIRS):
        # find a random other speaker 
        # (speaker is not the same as current speaker and the pair doesn't already exist in this order)
        ids2, dokid_anfnummers2 = choice(list(ids_to_dokanf.items()))
        while ids1[0] == ids2[0] or (ids1, ids2) in comparisons.keys():
            ids2, dokid_anfnummers2 = choice(list(ids_to_dokanf.items()))
        #
        # create a pair of speeches
        # (possible it exists in the reverse order though)
        comparisons[(ids1, ids2)] = get_pair(dokid_anfnummers1, dokid_anfnummers2)

# save to a dataframe
new_df = pd.DataFrame.from_dict(comparisons, columns=["danf1", "danf2"], orient="index")
new_df = new_df.reset_index()
new_df[["pair1", "pair2"]] = pd.DataFrame(new_df["index"].tolist(), index=new_df.index)
new_df[["intressent_id1", "age1"]] = pd.DataFrame(new_df["pair1"].tolist(), index=new_df.index)
new_df[["intressent_id2", "age2"]] = pd.DataFrame(new_df["pair2"].tolist(), index=new_df.index)
new_df.drop(["pair1", "pair2", "index"], axis=1, inplace=True) 
new_df["pair_1"] = new_df[["danf1", "danf2"]].apply(lambda x: (x.danf1, x.danf2), axis=1)
new_df.drop(["danf1", "danf2"], axis=1, inplace=True)

# new_df.to_parquet("../metadata/across_speaker_comparisons_for_within_age_comparisons.parquet")
new_df.to_parquet("../metadata/across_speaker_comparisons.parquet")


# skip starting age
# sub_comparisons = dict(filter(lambda x: x[0][0][1] != ids_to_min_age[x[0][0][0]], comparisons.items()))
# small_comparisons = dict()
# for iids, danf_pair in sub_comparisons.items():
#     iid_age1 = iids[0]
#     if [x[0] for x in small_comparisons.keys()].count(iid_age1) < NUM_PAIRS // 2:
#         small_comparisons[iids] = danf_pair

# save to a dataframe
# new_df = pd.DataFrame.from_dict(sub_comparisons, columns=["danf1", "danf2"], orient="index")
# new_df = new_df.reset_index()
# new_df[["pair1", "pair2"]] = pd.DataFrame(new_df["index"].tolist(), index=new_df.index)
# new_df[["intressent_id1", "age1"]] = pd.DataFrame(new_df["pair1"].tolist(), index=new_df.index)
# new_df[["intressent_id2", "age2"]] = pd.DataFrame(new_df["pair2"].tolist(), index=new_df.index)
# new_df.drop(["pair1", "pair2", "index"], axis=1, inplace=True) 
# new_df["pair_1"] = new_df[["danf1", "danf2"]].apply(lambda x: (x.danf1, x.danf2), axis=1)
# new_df.drop(["danf1", "danf2"], axis=1, inplace=True)

# new_df.to_parquet("../metadata/across_speaker_comparisons_for_across_age_comparisons.parquet")


# within speaker start age vs other ages
# create pairs for each speaker between their youngest age and any other age up to AGE_RANGE
NUM_PAIRS = 3
AGE_RANGE = 10

intressent_ids = set(ids[0] for ids in ids_to_dokanf)
diff_age_pairs = defaultdict(set)

# for each speaker, find their youngest age, generate speech pairs between that age+1 and any other age up to AGE_RANGE (including start_age)
# each pair has NUM_PAIRS speeches associated with them
# for now, some speakers have gap years due to a preprocessing error, so instead I choose to sort the speeches by age for each speaker
for iid in intressent_ids:
    sub_ids_to_dokanf = dict(filter(lambda x: x[0][0] == iid, ids_to_dokanf.items()))
    # sub_ids_to_dokanf = dict(sorted(sub_ids_to_dokanf.items(), key=lambda x: x[0][1]))
    start_age = ids_to_min_age[iid]
    dokid_anfnummers1 = ids_to_dokanf[(iid, start_age)]
    for sub_iid in sub_ids_to_dokanf.keys():
        # skip starting age
        # if ids_to_min_age[iid] == sub_iid[1]:
        #     continue
    # for age in range(start_age, start_age+AGE_RANGE+1):
        dokid_anfnummers2 = ids_to_dokanf[sub_iid]
        while len(diff_age_pairs[((iid, start_age), sub_iid)]) < NUM_PAIRS:
            diff_age_pairs[((iid, start_age), sub_iid)].add(get_pair(dokid_anfnummers1, dokid_anfnummers2))
            

# save to a dataframe
new_df = pd.DataFrame.from_dict(diff_age_pairs, columns=[f"pair_{i+1}" for i in range(NUM_PAIRS)], orient="index")
new_df = new_df.reset_index()
new_df[["sp1", "sp2"]] = pd.DataFrame(new_df["index"].tolist(), index=new_df.index)
new_df[["intressent_id1", "age1"]] = pd.DataFrame(new_df["sp1"].tolist(), index=new_df.index)
new_df[["intressent_id2", "age2"]] = pd.DataFrame(new_df["sp2"].tolist(), index=new_df.index)
new_df.drop(["sp1", "sp2", "index"], axis=1, inplace=True) 
new_df = new_df[["intressent_id1", "age1"] + ["intressent_id2", "age2"] + [f"pair_{i+1}" for i in range(NUM_PAIRS)]]

new_df.to_parquet("../metadata/within_speaker_across_age_comparisons.parquet")