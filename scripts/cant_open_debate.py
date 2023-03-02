from pydub import AudioSegment
import os
import pickle as pkl
from tqdm import tqdm

os.chdir("../metadata")
metadata_dir = os.getcwd()
audio_dir = "/data/datasets/riksdagen_anforanden/data/audio"

os.chdir(audio_dir)

cant_open = set()

for dir in tqdm(next(os.walk("."))[1]):
    audio_files = next(os.walk(dir))[2]
    for audio_file in audio_files:
        if ".mp3" in audio_file:
            try:
                sound = AudioSegment.from_mp3(audio_file)
            except:
                cant_open.add(dir)
                break
with open(os.path.join(metadata_dir, "cant_open.pkl"), "wb") as outfile:
    pkl.dump(cant_open, outfile)

print(cant_open)