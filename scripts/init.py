import os
import audio_data_builder as adb
import downloader as dr
import sys
sys.path.append("../lib")
import utils

utils.make_dir("../data")
utils.make_dir("../data/audio")
utils.make_dir("../data/frames")

dr.download_data("../data/csv/avspeech_train.csv", 0, 10, "../data/audio/speakers_train", "../data/frames/train", 3.0, wait_tasks=False)
dr.download_data("../data/csv/avspeech_test.csv", 0, 10, "../data/audio/speakers_test", "../data/frames/test", 3.0, wait_tasks=False)
dr.download_data("../data/csv/noise.csv", 0, 10, "../data/audio/noise", None, 3.0)

adb.build("../data/audio_train", "../data/audio/speakers_train", 1000, 2)
adb.build("../data/audio_test", "../data/audio/speakers_test", 500, 2)
