import pandas as pd
import os
from collections import defaultdict
from random import choice, sample
from sklearn.metrics.pairwise import cosine_similarity

# within speaker within age
NUM_PAIRS = 3

df = pd.read_parquet("../metadata/bucketed_speeches.parquet")

ids_to_dokanf = df.groupby(
    ["intressent_id", "age"]).agg(list)["dokid_anfnummer"].to_dict()

pairs = defaultdict(set)

# check that this doesn't contain doubles both ways!!!
# first pass
# make as many pairs as possible where the two speeches *don't* come from the same debate
for ids, dokid_anfnummers in ids_to_dokanf.items():
    tmp_dokid_anfnummers = dokid_anfnummers[:]
    dokids = set(i.split("_")[0] for i in tmp_dokid_anfnummers)
    while len(dokids) > 1:
        danf1 = choice(tmp_dokid_anfnummers)
        danf2 = choice(tmp_dokid_anfnummers)
        while danf1.split("_")[0] == danf2.split("_")[0]:
            danf2 = choice(tmp_dokid_anfnummers)
        _ = tmp_dokid_anfnummers.pop(tmp_dokid_anfnummers.index(danf1))
        _ = tmp_dokid_anfnummers.pop(tmp_dokid_anfnummers.index(danf2))
        pairs[ids].add(tuple(sorted([danf1, danf2])))
        if len(pairs[ids]) == NUM_PAIRS:
            break
        dokids = set([i.split("_")[0] for i in tmp_dokid_anfnummers])

print(set([len(pair) for pair in pairs.values()]))

# second pass (include speeches from the same debate)
for ids, dokid_anfnummers in ids_to_dokanf.items():
    while len(pairs[ids]) < NUM_PAIRS:
        new_pair = set()
        while len(new_pair) < 2:
            new_pair.add(choice(dokid_anfnummers))
        # new_pair = (choice(dokid_anfnummers), choice(dokid_anfnummers))
        pairs[ids].add(tuple(sorted(list(new_pair))))

print(set([len(pair) for pair in pairs.values()]))

# new_df = pd.DataFrame(pairs, columns=["intressent_id", "age"] + [f"column_{i+1}" for i in range(6)])
new_df = pd.DataFrame.from_dict(pairs, columns=[f"pair_{i+1}" for i in range(NUM_PAIRS)], orient="index")
new_df = new_df.reset_index()
new_df[["intressent_id", "age"]] = pd.DataFrame(new_df["index"].tolist(), index=new_df.index)
new_df = new_df.drop("index", axis=1)
new_df = new_df[["intressent_id", "age"] + [f"pair_{i+1}" for i in range(NUM_PAIRS)]]
print(new_df)

new_df.to_parquet("../metadata/within_speaker_within_age_comparisons.parquet")


# across speakers
# this takes forever 💀
if False:
    comparisons = dict()
    i = 0
    for ids1, dokid_anfnummers1 in ids_to_dokanf.items():
        if (i+1) % 20 == 0:
            print(i+1)
        for ids2, dokid_anfnummers2 in ids_to_dokanf.items():
            if ids1[0] == ids2[0]:
                continue
            dokids = set([i.split("_")[0] for i in dokid_anfnummers1] + [i.split("_")[0] for i in dokid_anfnummers2])
            danf1 = choice(dokid_anfnummers1)
            danf2 = choice(dokid_anfnummers2)
            if len(dokids) > 1:
                while danf1.split("_")[0] == danf2.split("_")[0]:
                    danf1 = choice(dokid_anfnummers1)
                    danf2 = choice(dokid_anfnummers2)
            pair = (danf1, danf2)
            while tuple(sorted(list(pair))) in set(comparisons.values()):
                danf1 = choice(dokid_anfnummers1)
                danf2 = choice(dokid_anfnummers2)
                if len(dokids) > 1:
                    while danf1.split("_")[0] == danf2.split("_")[0]:
                        danf1 = choice(dokid_anfnummers1)
                        danf2 = choice(dokid_anfnummers2)
                pair = (danf1, danf2)
            comparisons[(ids1, ids2)] = pair
        i += 1

    new_df = pd.DataFrame.from_dict(comparisons, columns=["danf1", "danf2"], orient="index")
    new_df = new_df.reset_index()
    new_df[["pair1", "pair2"]] = pd.DataFrame(new_df["index"].tolist(), index=new_df.index)
    new_df[["intressent_id1", "age1"]] = pd.DataFrame(new_df["pair1"].tolist(), index=new_df.index)
    new_df[["intressent_id2", "age2"]] = pd.DataFrame(new_df["pair2"].tolist(), index=new_df.index)
    new_df.drop(["pair1", "pair2", "index"], axis=1, inplace=True) 
    new_df["pair_1"] = new_df[["danf1", "danf2"]].apply(lambda x: (x.danf1, x.danf2), axis=1)
    new_df.drop(["danf1", "danf2"], axis=1, inplace=True)

    new_df.to_parquet("../metadata/across_speaker_comparisons.parquet")


# across speakers SMALL
# this does not take forever :D
comparisons = dict()
NUM_PAIRS = 6
# sorted_comparisons = set()
for ids1, dokid_anfnummers1 in ids_to_dokanf.items():
    for _ in range(NUM_PAIRS):
        ids2, dokid_anfnummers2 = choice(list(ids_to_dokanf.items()))
        while ids1[0] == ids2[0] or (ids1, ids2) in comparisons.keys():
            ids2, dokid_anfnummers2 = choice(list(ids_to_dokanf.items()))
        
        dokids = set([i.split("_")[0] for i in dokid_anfnummers1] + [i.split("_")[0] for i in dokid_anfnummers2])
        danf1 = choice(dokid_anfnummers1)
        danf2 = choice(dokid_anfnummers2)
        if len(dokids) > 1:
            while danf1.split("_")[0] == danf2.split("_")[0]:
                danf1 = choice(dokid_anfnummers1)
                danf2 = choice(dokid_anfnummers2)
        pair = (danf1, danf2)
        while pair in set(comparisons.values()):
            danf1 = choice(dokid_anfnummers1)
            danf2 = choice(dokid_anfnummers2)
            if len(dokids) > 1:
                while danf1.split("_")[0] == danf2.split("_")[0]:
                    danf1 = choice(dokid_anfnummers1)
                    danf2 = choice(dokid_anfnummers2)
            pair = (danf1, danf2)
        comparisons[(ids1, ids2)] = pair

new_df = pd.DataFrame.from_dict(comparisons, columns=["danf1", "danf2"], orient="index")
new_df = new_df.reset_index()
new_df[["pair1", "pair2"]] = pd.DataFrame(new_df["index"].tolist(), index=new_df.index)
new_df[["intressent_id1", "age1"]] = pd.DataFrame(new_df["pair1"].tolist(), index=new_df.index)
new_df[["intressent_id2", "age2"]] = pd.DataFrame(new_df["pair2"].tolist(), index=new_df.index)
new_df.drop(["pair1", "pair2", "index"], axis=1, inplace=True) 
new_df["pair_1"] = new_df[["danf1", "danf2"]].apply(lambda x: (x.danf1, x.danf2), axis=1)
new_df.drop(["danf1", "danf2"], axis=1, inplace=True)

new_df.to_parquet("../metadata/across_speaker_comparisons_small.parquet")

# within speaker start age vs other ages
NUM_PAIRS = 6
AGE_RANGE = 10

df = pd.read_parquet("../metadata/bucketed_speeches.parquet")

ids_to_dokanf = df.groupby(
    ["intressent_id", "age"]).agg(list)["dokid_anfnummer"].to_dict()

intressent_ids = set(ids[0] for ids in ids_to_dokanf)

pairs = defaultdict(set)

for iid in intressent_ids:
    sub_ids_to_dokanf = dict(filter(lambda x: x[0][0] == iid, ids_to_dokanf.items()))
    sub_ids_to_dokanf = dict(sorted(sub_ids_to_dokanf.items(), key=lambda x: x[0][1]))
    start_age = min([i[1] for i in sub_ids_to_dokanf.keys()])
    for sub_iid in sub_ids_to_dokanf.keys():
        if sub_iid[1] >= start_age + AGE_RANGE:
            break
    # for age in range(start_age, start_age+AGE_RANGE+1):
        for _ in range(NUM_PAIRS):
            danf1 = choice(ids_to_dokanf[(iid, start_age)])
            danf2 = choice(ids_to_dokanf[sub_iid])
            while danf1 == danf2 or (danf1, danf2) in pairs[((iid, start_age), sub_iid)]:
                danf1 = choice(ids_to_dokanf[(iid, start_age)])
                danf2 = choice(ids_to_dokanf[sub_iid])
            pairs[((iid, start_age), sub_iid)].add((danf1, danf2))

print(set([len(pair) for pair in pairs.values()]))

# new_df = pd.DataFrame(pairs, columns=["intressent_id", "age"] + [f"column_{i+1}" for i in range(6)])
new_df = pd.DataFrame.from_dict(pairs, columns=[f"pair_{i+1}" for i in range(NUM_PAIRS)], orient="index")
new_df = new_df.reset_index()
new_df[["sp1", "sp2"]] = pd.DataFrame(new_df["index"].tolist(), index=new_df.index)
new_df[["intressent_id1", "age1"]] = pd.DataFrame(new_df["sp1"].tolist(), index=new_df.index)
new_df[["intressent_id2", "age2"]] = pd.DataFrame(new_df["sp2"].tolist(), index=new_df.index)
new_df.drop(["sp1", "sp2", "index"], axis=1, inplace=True) 
new_df = new_df[["intressent_id1", "age1"] + ["intressent_id2", "age2"] + [f"pair_{i+1}" for i in range(NUM_PAIRS)]]

new_df.to_parquet("../metadata/within_speaker_across_age_comparisons.parquet")

subset = set(list(reduce(lambda x, y: list(x) + list(y), pairs.keys()))) # for if I want to keep only the iids+age from the previous step

sub_comparisons = dict(filter(lambda x: x[0][0] in subset, comparisons.items()))

new_df = pd.DataFrame.from_dict(sub_comparisons, columns=["danf1", "danf2"], orient="index")
new_df = new_df.reset_index()
new_df[["pair1", "pair2"]] = pd.DataFrame(new_df["index"].tolist(), index=new_df.index)
new_df[["intressent_id1", "age1"]] = pd.DataFrame(new_df["pair1"].tolist(), index=new_df.index)
new_df[["intressent_id2", "age2"]] = pd.DataFrame(new_df["pair2"].tolist(), index=new_df.index)
new_df.drop(["pair1", "pair2", "index"], axis=1, inplace=True) 
new_df["pair_1"] = new_df[["danf1", "danf2"]].apply(lambda x: (x.danf1, x.danf2), axis=1)
new_df.drop(["danf1", "danf2"], axis=1, inplace=True)

new_df.to_parquet("../metadata/across_speaker_comparisons_small.parquet")