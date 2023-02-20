import pandas as pd
import os
from collections import defaultdict
from random import choice

df = pd.read_parquet("../metadata/bucketed_speeches.parquet")

ids_to_dokanf = df.groupby(
    ["intressent_id", "age"]).agg(list)["dokid_anfnummer"].to_dict()

pairs = defaultdict(set)

for ids, dokid_anfnummers in ids_to_dokanf.items():
    while len(pairs[ids]) < 27:
        new_pair = (choice(dokid_anfnummers), choice(dokid_anfnummers))
        if (new_pair[0] != new_pair[1]) and new_pair not in pairs[ids]:
            pairs[ids].add(new_pair)
