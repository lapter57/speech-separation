import os
import sys
sys.path.append("../lib")
import utils

from config import Config
from downloader import Downloader
from data_builder import DataBuilder

config = Config("/usr/dev/speech-separation/config.yaml")
#downloader = Downloader(config)
data_builder = DataBuilder(config, config.data.num_workers * 2)

#utils.make_dirs("../data")
#utils.make_dirs("../data/audio")
#utils.make_dirs("../data/frames")

# Download data
#downloader.download_data("../data/csv/avspeech_train.csv", 0, 55000, wait_tasks=False)
#downloader.download_data("../data/csv/avspeech_test.csv", 0, 1000)
#downloader.download_noise_data("../data/csv/noise.csv", 0, 1000)

# Build data
#data_builder.build_embs(wait_tasks=False)
#data_builder.build_embs(is_train=False)

data_builder.build_audio(38000, wait_tasks=False)
data_builder.build_audio(1000, is_train=False)
