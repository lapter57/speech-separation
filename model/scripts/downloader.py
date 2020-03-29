import sys
sys.path.append("../lib")
import avhandler as avh
import utils
import pandas as pd
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=5)

def make_dirs(audio_path, frames_path, 
              remake_audio_dir=False, remake_frames_dir=False):
    utils.make_dir(audio_path, remake_audio_dir)
    utils.make_dir(frames_path, remake_frames_dir)

def download_data(csv_path, start_idx, end_idx, 
                  audio_path, frames_path, length=None,
                  fps=25, video_ext="mp4", audio_ext="wav", 
                  sample_rate=16000, remake_audio_dir=False,
                  remake_frames_dir=False, wait_tasks=True):
    df = pd.read_csv(csv_path,
                     usecols=[0,1,2],
                     names=["youtube_id", "start_time", "end_time"],
                     comment="#")
    make_dirs(audio_path, frames_path, remake_audio_dir, remake_frames_dir)
    id = start_idx
    for i in range(start_idx, end_idx):
        youtube_id = df.loc[i, "youtube_id"]
        start_time = float(df.loc[i, "start_time"])
        end_time = float(df.loc[i, "end_time"]) if length == None else start_time + length
        filename = str(id) + ":" + youtube_id
        executor.submit(avh.download_data, youtube_id, filename, start_time, end_time, audio_path, frames_path)
        id += 1
    if wait_tasks:
        executor.shutdown(wait=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--csv_path", action="store",
                        dest="csv_path", required=True,
                        help="Path to a csv file whose first three columns is: youtube_id, start_time, end_time")
    parser.add_argument("--audio_path", action="store",
                        dest="audio_path",
                        help="Path to the folder where audio data will be saved")
    parser.add_argument("--frames_path", action="store",
                        dest="frames_path",
                        help="Path to the folder where frames of video will be saved.")
    parser.add_argument("--fps", action="store", type=int, 
                        dest="fps", default=25, help="Fps")
    parser.add_argument("--audio_ext", action="store", 
                        dest="audio_ext", default="wav", help="Audio extension (default=wav)")
    parser.add_argument("--video_ext", action="store", 
                        dest="video_ext", default="mp4", help="Video extension (default=mp4)")
    parser.add_argument("--start", action="store", type=int,
                        dest="start_idx", default=0, 
                        help="Start index (default=0)")
    parser.add_argument("--end", action="store", type=int, dest="end_idx",
                        default=10, help="End index (default=10)")
    parser.add_argument("--length", action="store", type=float, dest="length",
                        default=3.0, help="Audio duration(default=3.0)")
    args = parser.parse_args()
    download_data(args.csv_path, args.start_idx, args.end_idx,
                  args.audio_path, args.frames_path, args.length)