import wget
import os
import audio_data_builder as adb
import audio_downloader as ad

TRAIN_CSV_URL = "https://storage.cloud.google.com/avspeech-files/avspeech_train.csv"
TEST_CSV_URL = "https://storage.cloud.google.com/avspeech-files/avspeech_test.csv"
NOISE_CSV_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv"

adb.init_dir("../data")
adb.init_dir("../data/csv")
adb.init_dir("../data/audio")

wget.download(TRAIN_CSV_URL, '../data/csv/avspeech_train.csv')
wget.download(TEST_CSV_URL, '../data/csv/avspeech_test.csv')
wget.download(NOISE_CSV_URL, '../data/csv/noise.csv')

ad.download_audio("../data/audio/speakers_train", 0, 1000, 3.0)
ad.download_audio("../data/audio/speakers_test", 0, 500, 3.0)
ad.download_audio("../data/audio/noise", 0, 500, 3.0)

adb.build("../data/audio_train", "../data/audio/speakers_train", 
          1000, 2, "../data/audio/noise")
adb.build("../data/audio_test", "../data/audio/speakers_test", 
          500, 2, "../data/audio/noise")