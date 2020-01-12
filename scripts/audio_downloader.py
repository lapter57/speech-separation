import sys
sys.path.append("../lib")
import AVHandler as avh
import os
import pandas as pd
from argparse import ArgumentParser

def download_audio(path, start_idx, end_idx, length=None):
    df = pd.read_csv(args.path,
                     usecols=[0,1,2],
                     names=["youtube_id", "start_time", "end_time"],
                     comment="#")
    try:
        os.mkdir(args.dest)
    except OSError:
        pass
    id = 0
    for i in range(start_idx, end_idx):
        youtube_id = df.loc[i,"youtube_id"]
        start_time = float(df.loc[i,"start_time"])
        end_time = start_time + length
        if length == None:
            end_time = float(df.loc[i,"end_time"])
        filename = str(id) + ":" + youtube_id
        if avh.download_audio(youtube_id, filename, path):
            id += 1
        avh.cut_audio(youtube_id, start_time, end_time, filename, path, with_remove=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", action="store",
                        dest="path", required=True,
                        help="Path to a csv file whose first three columns is: youtube_id, start_time, end_time")
    parser.add_argument("--dest", action="store",
                        dest="dest", required=True,
                        help="Path to the folder where audio data will be saved")
    parser.add_argument("--start", action="store", type=int,
                        dest="start_idx", default=0, 
                        help="Start index (default=0)")
    parser.add_argument("--end", action="store", type=int, dest="end_idx",
                        default=10, help="End index (default=10)")
    parser.add_argument("--length", action="store", type=float, dest="length",
                        default=3.0, help="Audio duration(default=3.0)")
    args = parser.parse_args()
    download_audio(args.dest, args.start_idx, args.end_idx, args.length)
