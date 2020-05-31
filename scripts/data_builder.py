import sys
sys.path.append("../lib")
import utils
import os
import numpy as np
import librosa
import insightface
import cv2
import shutil
import math
import itertools

from tqdm import tqdm
from audio import Audio
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, wait

DIR = ("{path}/{dirname}")

class DataBuilder():
    def __init__(self, config, cpu_count=os.cpu_count()):
        self.config = config
        self.executor = ThreadPoolExecutor(cpu_count)
        self.fs = []
        self.audio_handler = Audio(config)
        self.face_model = self.prepare_face_model()

    def prepare_face_model(self):
        face_model = insightface.app.FaceAnalysis()
        face_model.prepare(ctx_id=self.config.face.ctx_id, nms=self.config.face.nms)
        return face_model

    def init_audio_dirs(self, path):
        utils.make_dirs(path)
        mix_path = DIR.format(path=path, dirname="mix")
        clean_path = DIR.format(path=path, dirname="clean")
        utils.make_dirs(mix_path)
        utils.make_dirs(clean_path)
        return mix_path, clean_path

    def remove_short_audios(self, speech_paths):
        for i, s in enumerate(speech_paths):
            a, _ = librosa.load(s, sr=self.config.audio.sample_rate)
            if a.shape[0] < self.config.audio.sample_rate * self.config.audio.len:
                speech_paths = np.delete(speech_paths, i)
        return speech_paths

    def split_with_cross_product(self, speech_paths):
        splited_speech = np.array_split(speech_paths, self.config.data.num_speakers)
        cross_product = itertools.product(*splited_speech)
        return np.array([x for x in cross_product])
    
    def divide_batches(self, arr, n):
        for i in range(0, len(arr), n):
            yield arr[i:i+n]
    
    def split_with_seq(self, speech_paths):
        splited = np.array([x for x in self.divide_batches(speech_paths, self.config.data.num_speakers)])
        return np.array([x for x in splited if x.size == self.config.data.num_speakers])
    
    def split_speech(self, speech_paths):
        switcher = {
            "cp": lambda: self.split_with_cross_product(speech_paths),
            "seq": lambda: self.split_with_seq(speech_paths),
        }
        return switcher.get(self.config.data.build_mode)()

    def add_noise(self, mix, noise_paths):
        noise_path = random.choice(noise_paths)
        noise, _ = librosa.load(noise_path, sr=self.config.audio.sample_rate)
        L = int(self.config.audio.sample_rate * self.config.audio.len)
        noise = noise[:L]
        mix += noise
        return mix

    def prepare_speech_data(self, speech_path, audio_data_path, usage):
        mix_path, clean_path = self.init_audio_dirs(audio_data_path)
        speech_paths = utils.get_files(speech_path)
        np.random.shuffle(speech_paths)
        speech_paths = speech_paths[:usage]
        speech_paths = self.remove_short_audios(speech_paths)
        return speech_paths, mix_path, clean_path

    def start_build_audio(self, speech_paths, mix_path, 
                          clean_path, noise_paths=None):
        for paths in tqdm(self.split_speech(speech_paths), desc="Data building"):
            try:
                cleans = []
                for p in paths:
                    w, _ = librosa.load(p, sr=self.config.audio.sample_rate)
                    L = int(self.config.audio.sample_rate * self.config.audio.len)
                    w = w[:L]
                    cleans.append([p, w])

                mix = None
                for i, w in enumerate(cleans):
                    if i == 0:
                        mix = list(w[1])
                    else:
                        mix += w[1]

                if noise_paths is not None:
                    mix = self.add_noise(mix, noise_paths)
            
                norm = np.max(np.abs(mix)) * 1.1
                mix /= norm
                for i in range(len(cleans)):
                    cleans[i][1] = cleans[i][1] / norm

                mix_filename = ""
                for p in paths:
                    mix_filename += utils.basename(p) + "."
                np.save(("{}/{}npy").format(mix_path, mix_filename), self.audio_handler.wav2spec(mix))
    
                for w in cleans:
                    clean_filename = utils.basename(w[0]) + ".npy"
                    np.save(("{}/{}").format(clean_path, clean_filename), self.audio_handler.wav2spec(w[1]))
    
            except Exception:
                try:
                    os.remove(("{}/{}npy").format(mix_path, mix_filename))
                    print("[ERROR] remove " + ("{}/{}npy").format(mix_path, mix_filename))
                except Exception:
                    pass
        

    def build_audio(self, usage=2, is_train=True, with_noise=False, wait_tasks=True):
        audio_target_dir = 'train' if is_train else 'test' 
        speech_path = os.path.join(self.config.data.audio.path, 
                                   self.config.data.audio.speech_dir, 
                                   audio_target_dir) 
        audio_data_path = os.path.join(self.config.data.audio.path, audio_target_dir)
        noise_paths = utils.get_files(os.path.join(self.config.data.audio.path, self.config.data.audio.noise_dir)) if with_noise else None
        speech_paths, mix_path, clean_path = self.prepare_speech_data(speech_path, audio_data_path, usage)

        paths = np.array_split(speech_paths, self.config.data.num_workers)
        for p in paths:
            self.fs.append(self.executor.submit(self.start_build_audio, p, mix_path, 
                                                clean_path, noise_paths))
        if wait_tasks:
            wait(self.fs)
            self.fs.clear()

    def face_detect(self, image_path, model=None):
        model = self.face_model if model is None else model
        img = cv2.imread(image_path)
        faces = model.get(img)
        if len(faces) != 0:
            box = faces[0].bbox.astype(np.int).flatten()
            box[box < 0] = 0
            crop_img = img[box[1]:box[3], box[0]:box[2]]
            crop_img = cv2.resize(crop_img, (112, 112))
            emb = np.array(faces[0].normed_embedding.tolist()) 
            return emb
        return np.zeros((1, self.config.face.emb_size))
    
    def process_frames(self, ids, frames_path, emb_path, remove_frames=True, use_new_model=False):
        model = self.prepare_face_model() if use_new_model else self.face_model 
        for id in tqdm(ids, desc="Processing frames"):
            found_files = [name for name in os.listdir(frames_path) if name.startswith(str(id) + ":")]
            if (len(found_files) != 0):
                file = found_files[0]
                prefix_name = ":".join(file.split(":", 2)[:2])
                embs = np.zeros((self.config.face.num_faces, 1, self.config.face.emb_size))
                for j in range(1, self.config.face.num_faces + 1):
                    filename = prefix_name + ":{:0>2d}.jpg".format(j)
                    embs[j - 1, : ] = face_detect(os.path.join(frames_path, filename), model)
                np.save(os.path.join(emb_path, "{}.npy".format(prefix_name)), embs)
            if remove_frames:
                for f in found_files:
                    os.remove(os.path.join(frames_path, f))


    def build_embs(self, is_train=True, remove_frames=True, wait_tasks=True):
        target_dir = "train" if is_train else "test"
        frames_path = os.path.join(self.config.data.video.frames_path, target_dir)
        emb_path = os.path.join(self.config.data.video.emb_path, target_dir)
        utils.make_dirs(emb_path)
    
        ids = list(set([int(name.split(":")[0]) for name in os.listdir(frames_path)]))
        ids = np.array_split(ids, self.config.face.num_workers)
        for i in ids:
            self.fs.append(self.executor.submit(self.process_frames, 
                                                i, frames_path, emb_path, 
                                                remove_frames, False if i == 0 else True))
        if wait_tasks:
            wait(self.fs)
            self.fs.clear()
  
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", action="store",
                        dest="config", required=True,
                        help="Path to the config file")
    parser.add_argument("--train", action="store",
                        dest="is_train", default=True, type=bool,
                        help="Build a train data (default=true)")
    parser.add_argument("--noise", action="store",
                        dest="with_noise", default=False, type=bool,
                        help="Build an audio data with noise (default=false)")
    parser.add_argument("--usage", action="store", type=int,
                        dest="usage", default=2,
                        help="Data usage to generate audio data (default=2)")
    args = parser.parse_args()
    config = Config(args.config)
    data_builder = DataBuilder(config)
    data_builder.build_audio(args.usage, args.is_train, args.with_noise)
    data_builder.build_embs(args.is_train)

