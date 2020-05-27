import sys
sys.path.append("../lib")
import os
import avhandler as avh
import utils
import pandas as pd
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, wait


class Downloader():
    def __init__(self, config, cpu_count=os.cpu_count()):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=cpu_count)
        self.fs = []

    def make_dirs(self, audio_path, frames_path, 
                  remake_audio_dir=False, remake_frames_dir=False):
        utils.make_dirs(audio_path, remake_audio_dir)
        utils.make_dirs(frames_path, remake_frames_dir)

    def download(self, csv_path, start_idx,
                 end_idx, target_dir, with_frames=True,
                 remake_audio_dir=False, remake_frames_dir=False,
                 wait_tasks=True):
        df = pd.read_csv(csv_path,
                         usecols=[0,1,2],
                         names=["youtube_id", "start_time", "end_time"],
                         comment="#")
        audio_dir = os.path.join(self.config.data.audio.path, target_dir)
        frame_dir = os.path.join(self.config.data.video.frames_path, target_dir) if with_frames else None
        make_dirs(audio_dir, frame_dir, remake_audio_dir, remake_frames_dir)
        id = start_idx
        for i in range(start_idx, end_idx):
            youtube_id = df.loc[i, "youtube_id"]
            start_time = float(df.loc[i, "start_time"])
            end_time = float(df.loc[i, "end_time"]) if self.config.audio.len == None else start_time + self.config.audio.len
            filename = str(id) + ":" + youtube_id
            self.fs.append(self.executor.submit(avh.download_data, youtube_id,
                                                filename, start_time, end_time, 
                                                audio_dir, frame_dir))
            id += 1
        if wait_tasks:
            wait(self.fs)
            self.fs.clear()
    
    def download_data(self, csv_path, start_idx, 
                      end_idx, is_train=True, 
                      with_frames=True, remake_audio_dir=False,
                      remake_frames_dir=False, wait_tasks=True):
        target_dir_name = 'train' if is_train else 'test'
        target_dir = os.path.join(self.config.data.audio.speech_dir, target_dir_name)
        download(csv_path, start_idx, end_idx, target_dir, 
                 with_frames, remake_audio_dir, remake_frames_dir,
                 wait_tasks)

    def download_noise_data(self, csv_path, start_idx,  
                            end_idx, with_frames=False, remake_audio_dir=False,
                            remake_frames_dir=False, wait_tasks=True):
        download(csv_path, start_idx, end_idx, self.config.data.audio.noise_dir, 
                 with_frames, remake_audio_dir, remake_frames_dir,
                 wait_tasks)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", action="store",
                        dest="config", required=True,
                        help="Path to a config file")
    parser.add_argument("--csv_path", action="store",
                        dest="csv_path", required=True,
                        help="Path to a csv file whose first three columns is: youtube_id, start_time, end_time")
    parser.add_argument("--start", action="store", type=int,
                        dest="start_idx", default=0,
                        help="Start index (default=0)")
    parser.add_argument("--end", action="store", type=int, dest="end_idx",
                        default=10, help="End index (default=10)")
    parser.add_argument("--target", action="store", dest="target_dir",
                        required=True, help="Name of target dir")
    parser.add_argument("--frames", action="store", type=bool, dest="with_frames",
                        default=True, help="Downloading with frames (default=true)")
    args = parser.parse_args()
    config = Config(args.config)
    Downloader(config).download(args.csv_path, args.start_idx, args.end_idx, args.target_dir, args.with_frames)
