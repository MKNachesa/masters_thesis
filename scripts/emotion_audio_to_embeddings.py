import nemo.collections.asr as nemo_asr
import os
from tqdm import tqdm
import pickle as pkl

# this script only works if you have the emotional speech dataset on your disk
# link: https://hltsingapore.github.io/ESD/index.html
# feel free to change "data_dir" so it works of course

data_dir = "../../../Research and Development/Project Speech Emotion Recognition/Emotional Speech Dataset (ESD)"
save_dir = "../data/emotion_voiceprints"

speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')

voiceprints = {"en": dict(), "ch": dict()}

with tqdm(sorted(next(os.walk(data_dir))[1]), total=20*4*350) as speakers:
    for speaker in speakers:
        # print(f"Processing speaker {speaker}")
        if 1 <= int(speaker) <= 10:
            lang = "ch"
        elif 11 <= int(speaker) <= 20:
            lang = "en"
        else:
            raise ValueError(f"Directory {speaker} does not exist")
        speaker_dir = os.path.join(data_dir, speaker)
        speaker_dict = dict()
        for emotion in next(os.walk(speaker_dir))[1]:
            emotion_dir = os.path.join(speaker_dir, emotion)
            emotion_dict = dict()
            for splt in next(os.walk(emotion_dir))[1]:
                splt_path = os.path.join(emotion_dir, splt)
                splt_list = []

                speakers.set_postfix(speaker=speaker, emotion=emotion, split=splt)
                for sound_file in list(filter(lambda x: ")" not in x, next(os.walk(splt_path))[2])):#,
                                        # desc=f"Emotion: {emotion:<7} split: {splt:<10}"): # REMOVE SPLICE LATER
                    sound_path = os.path.join(splt_path, sound_file)
                    emb = speaker_model.get_embedding(sound_path).cpu()
                    splt_list.append(emb)
                    speakers.update(1)
                emotion_dict[splt] = splt_list
            speaker_dict[emotion] = emotion_dict
        voiceprints[lang][speaker] = speaker_dict


total_len = 0
for lang, lang_value in voiceprints.items():
    for speaker, speaker_value in lang_value.items():
        for emotion, emotion_value in speaker_value.items():
            for split, split_data in emotion_value.items():
                total_len += len(split_data)

print(f"There are {total_len} voiceprints.")
if total_len == 20*4*350: # num_speakers * num_emotions * num_utterances
    print("This is expected")
else:
    print(f"This is unexpected. There should be {20*4*350} voiceprints")

with open(os.path.join(save_dir, "emotion_voiceprints.pkl"), "wb") as f:
    pkl.dump(voiceprints, f)