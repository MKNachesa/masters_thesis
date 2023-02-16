#titanet demo
from omegaconf import OmegaConf
import os
import nemo.collections.asr as nemo_asr
from torch.nn import CosineSimilarity
import pandas as pd
import pickle as pkl

cos = CosineSimilarity(dim=1, eps=1e-8)
speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')

os.chdir("../metadata")
data = os.getcwd()
os.chdir("../../riksdagen_anforanden/data/audio")
audio_data = os.getcwd()
os.chdir("../../../masters_thesis/scripts")

filtered_path = os.path.join(data, "filtered_speeches.parquet")
df_filt = pd.read_parquet(filtered_path)

df_filt["filename_anforande_audio"] = df_filt[["filename", "start_segment", "duration_segment"]].apply(lambda x: f"{os.path.join(audio_data, x.filename.strip('.wav'))}_{str(x.start_segment)}_{str(x.start_segment+x.duration_segment)}.wav", axis=1)

sp1 = df_filt[df_filt.dokid == "GR01LU22"]["filename_anforande_audio"].iloc[0]
sp2 = df_filt[df_filt.dokid == "GR01LU22"]["filename_anforande_audio"].iloc[1]

sp1 = '/home/mayanachesa/Documents/Thesis/riksdagen_anforanden/data/audio/GR10529/2442210130028245921_aud_463761.3573528429_523761.3573528429.wav'
sp2 = '/home/mayanachesa/Documents/Thesis/riksdagen_anforanden/data/audio/GR10529/2442210130028245921_aud_791053.5671805934_851053.5671805934.wav'

# NEMO_ROOT = os.getcwd()
# MODEL_CONFIG = os.path.join(NEMO_ROOT,'../conf/titanet-large.yaml')
# config = OmegaConf.load(MODEL_CONFIG)

emb1 = speaker_model.get_embedding(sp1)
emb2 = speaker_model.get_embedding(sp2)

pkl.dump({"emb1": emb1, "emb2": emb2}, open("../metadata/embeddings_test.pkl", "wb"))

print(cos(emb1, emb2))