import os
import audio_data_builder as adb
import downloader as dr
import face_detection as fd
import sys
sys.path.append("../lib")
import utils

utils.make_dirs("../data")
utils.make_dirs("../data/audio")
utils.make_dirs("../data/frames")

# Download data
dr.download_data("../data/csv/avspeech_train.csv", 0, 10, "../data/audio/speakers_train", "../data/frames/train", 3.0, wait_tasks=False)
dr.download_data("../data/csv/avspeech_test.csv", 0, 10, "../data/audio/speakers_test", "../data/frames/test", 3.0, wait_tasks=False)
dr.download_data("../data/csv/noise.csv", 0, 10, "../data/audio/noise", None, 3.0)

# Build data
fd.save_embeddings("../data/frames/train", "../data/emb/train")
fd.save_embeddings("../data/frames/test", "../data/emb/test")

adb.build("../data/audio_train", "../data/audio/speakers_train", 10, 2, background=True)
adb.build("../data/audio_test", "../data/audio/speakers_test", 10, 2)
