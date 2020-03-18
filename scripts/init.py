import wget
import os
import audio_data_builder as adb
import downloader as dr
import sys
sys.path.append("../lib")
import utils

utils.make_dir("../data")
utils.make_dir("../data/csv")
utils.make_dir("../data/audio")
utils.make_dir("../data/frames")

dr.download_data("../data/csv/avspeech_train.csv", 0, 1000, "../data/audio/speakers_train", "../data/frames", 3.0)
dr.download_data("../data/csv/avspeech_test.csv", 0, 500, "../data/audio/speakers_test", "../data/frames", 3.0)
dr.download_data("../data/csv/noise.csv", 0, 500, "../data/audio/noise", None, 3.0)

adb.build("../data/audio_train", "../data/audio/speakers_train", 1000, 2, "../data/audio/noise")
adb.build("../data/audio_test", "../data/audio/speakers_test", 500, 2, "../data/audio/noise")