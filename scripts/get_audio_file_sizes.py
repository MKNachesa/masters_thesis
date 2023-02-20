import os

os.chdir("../../riksdagen_anforanden/data/audio")

total_size = 0

for dir in next(os.walk("."))[1]:
  audio_files = next(os.walk(dir))[2]
  for audio_file in audio_files:
    # if ".wav" in audio_file:
    total_size += os.path.getsize(os.path.join(dir, audio_file))

print(total_size//1024/1024//1024)