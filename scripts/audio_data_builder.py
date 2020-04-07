import sys
sys.path.append("../lib")
import utils
import avhandler as avh
import os
import numpy as np
import librosa
import itertools
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=5)

DIR = ("{path}/{dirname}")

def init_dirs(path):
    utils.make_dirs(path)
    mix_path = DIR.format(path=path, dirname="mix")
    clean_path = DIR.format(path=path, dirname="clean")
    crm_path = DIR.format(path=path, dirname="crm")
    utils.make_dirs(mix_path)
    utils.make_dirs(clean_path)
    utils.make_dirs(crm_path)
    return mix_path, clean_path, crm_path

def build_clean_data(clean_path, speaker_paths):
    for path in speaker_paths:
        audio, sr = librosa.load(path, sr=16000)
        audio = utils.stft(audio)
        filename = utils.basename(path)
        np.save(("{}/{}.npy").format(clean_path, filename), audio)

def split_with_cross_product(speaker_paths, num_speakers=2):
    split_speakers = np.array_split(speaker_paths, num_speakers)
    cross_product = itertools.product(*split_speakers)
    return np.array([x for x in cross_product])

def divide_batches(arr, n):
    for i in range(0, len(arr), n):
        yield arr[i:i+n]

def split_with_seq(speaker_paths, num_speakers=2):
    split = np.array([x for x in divide_batches(speaker_paths, num_speakers)])
    return np.array([x for x in split if x.size == num_speakers])

def split_speakers(speaker_paths, num_speakers=2, mode="cp"):
    switcher = {
        "cp": lambda: split_with_cross_product(speaker_paths, num_speakers),
        "seq": lambda: split_with_seq(speaker_paths, num_speakers),
    }
    return switcher.get(mode)()

def build_mix_data(mix_path, speaker_paths, num_speakers=2, noise_path=None, mode="cp"):
    for paths in split_speakers(speaker_paths, num_speakers, mode):
        mix, noise_file = avh.mix_audio(paths, noise_path)
        mix = utils.stft(mix)
        filename = ""
        for path in paths:
            filename += utils.basename(path) + "."
        if noise_path:
            filename += utils.NOISE_PREFIX + utils.basename(noise_file) + "."
        np.save(("{}/{}npy").format(mix_path, filename), mix)

def build_crm_data(crm_path, mix_path, clean_path, batch_size = 100):
    mix_files = utils.get_files(mix_path)
    clean_files = utils.get_files(clean_path)
    if len(mix_files) < batch_size:
        batch_size = 1
    mix_files = np.array_split(mix_files, batch_size)
    for mf in mix_files:
        mix_npys = np.array([(utils.basename(f), np.load(f)) for f in mf])
        for mix_npy in mix_npys:
            clean_filenames = utils.get_clean_in_mix(mix_npy[0])
            for clean_filename in clean_filenames:
                clean_file = utils.find_paths_contains(clean_filename, clean_files)[0]
                clean_audio = np.load(clean_file)
                cRM = utils.cRM(clean_audio, mix_npy[1])
                filename = ("clean:{} mix:{}").format(clean_filename, mix_npy[0])
                np.save(("{}/{}.npy").format(crm_path, filename), cRM)

def start_build(path, speaker_path, usage=2, 
                num_speakers=2, noise_path=None, mode="seq"):
    mix_path, clean_path, crm_path = init_dirs(path)
    speaker_paths = utils.get_files(speaker_path)
    np.random.shuffle(speaker_paths)
    speaker_paths = speaker_paths[:usage]
    build_clean_data(clean_path, speaker_paths)
    print("Clean data was builded")
    build_mix_data(mix_path, speaker_paths, num_speakers, noise_path, mode)
    print("Mix data was builded")
    build_crm_data(crm_path, mix_path, clean_path)
    print("Crm data was builded")
 

def build(path, speaker_path, usage=2, 
          num_speakers=2, noise_path=None, 
          mode="seq", background=False):
    if (background):
        executor.submit(start_build, path, speaker_path, usage, num_speakers, noise_path, mode)
    else:
        start_build(path, speaker_path, usage, num_speakers, noise_path, mode)
   
if __name__ == "__main__":
    modes=["cp", "seq"]
    parser = ArgumentParser()
    parser.add_argument("--path", action="store",
                        dest="path", required=True,
                        help="Path to the folder where audio data will be saved")
    parser.add_argument("--speakers", action="store",
                        dest="speaker_path", required=True,
                        help="Path to the folder where the audio data of the speakers is stored")
    parser.add_argument("--usage", action="store", type=int,
                        dest="usage", default=2,
                        help="Data usage to generate audio data (default=2)")
    parser.add_argument("--nums", action="store", type=int,
                        dest="num_speakers", default=2,
                        help="Number of speakers used in the mix (default=2)")
    parser.add_argument("--noise", action="store",
                        dest="noise_path",
                        help="Path to the folder where the audio data of the noise is stored. If specified, noise will be used during build")
    parser.add_argument("--mode", choices=modes, action="store", 
                        dest="mode", default="seq", 
                        help="Generation mode for mix data (default=seq)")
    args = parser.parse_args()
    
    build(args.path, args.speaker_path, args.usage,
          args.num_speakers, args.noise_path, args.mode)
