import os

os.chdir("../../riksdagen_anforanden/data/audio")

for dir in next(os.walk("."))[1]:
  audio_files = next(os.walk(dir))[2]
  for audio_file in audio_files:
    if ".wav" in audio_file:
      os.remove(os.path.join(dir, audio_file))
