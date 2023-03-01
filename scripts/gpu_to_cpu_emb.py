import os
import pickle as pkl

scripts_dir = os.getcwd()
os.chdir("..")
thesis_dir = os.getcwd()
os.chdir("scripts")
vp_dir = os.path.join(thesis_dir, "data")
os.chdir(scripts_dir)
full_dir = os.path.join(vp_dir, "full")

for file in next(os.walk(full_dir))[2]:
    file_path = os.path.join(full_dir, file)
    with open(file_path, "rb") as infile:
        f_dict = pkl.load(infile)
    f_dict = dict(map(lambda x: (x[0], x[1].cpu()), f_dict.items()))
    with open(file_path, "wb") as outfile:
        pkl.dump(f_dict, outfile)