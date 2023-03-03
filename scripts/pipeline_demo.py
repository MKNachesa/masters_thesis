import pandas as pd
import os
from collections import defaultdict
from random import choice, sample
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_parquet("../metadata/bucketed_speeches.parquet")

ids_to_dokanf = df.groupby(
    ["intressent_id", "age"]).agg(list)["dokid_anfnummer"].to_dict()

pairs = defaultdict(set)

# check that this doesn't contain doubles both ways!!!
# first pass
# make as many pairs as possible where the two speeches *don't* come from the same debate
for ids, dokid_anfnummers in ids_to_dokanf.items():
    for danf1 in sample(dokid_anfnummers, len(dokid_anfnummers)):
        for danf2 in sample(dokid_anfnummers, len(dokid_anfnummers)):
            if danf1.split("_")[0] != danf2.split("_")[0]:
                pairs[ids].add((danf1, danf2))
            if len(pairs[ids]) == 6:
                break
        if len(pairs[ids]) == 6:
            break

print(set([len(pair) for pair in pairs.values()]))

# second pass (include speeches from the same debate)
for ids, dokid_anfnummers in ids_to_dokanf.items():
    while len(pairs[ids]) < 6:
        new_pair = (choice(dokid_anfnummers), choice(dokid_anfnummers))
        if (new_pair[0] != new_pair[1]) and new_pair not in pairs[ids]:
            pairs[ids].add(new_pair)

print(set([len(pair) for pair in pairs.values()]))

# new_df = pd.DataFrame(pairs, columns=["intressent_id", "age"] + [f"column_{i+1}" for i in range(6)])
new_df = pd.DataFrame.from_dict(pairs, columns=[f"pair_{i+1}" for i in range(6)], orient="index")
new_df = new_df.reset_index()
new_df[["intressent_id", "age"]] = pd.DataFrame(new_df["index"].tolist(), index=new_df.index)
new_df = new_df.drop("index", axis=1)
new_df = new_df[["intressent_id", "age"] + [f"pair_{i+1}" for i in range(6)]]
print(new_df)

new_df.to_parquet("../metadata/within_speaker_within_age_comparisons.parquet")