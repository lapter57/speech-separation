import os
import librosa
import tempfile
import numpy as np
import shutil
import random

DL_VIDEO = ("ffmpeg -i $(youtube-dl -f \"mp4\" --get-url {url}) -ss {start_time} -to {end_time} {file} </dev/null > /dev/null 2>&1 ;")
EXTRACT_AUDIO = ("ffmpeg -i {video_file} -f {audio_ext} -ar {sample_rate} -ac 1 -vn {audio_file} </dev/null > /dev/null 2>&1;")
CUT_AUDIO = ("sox {file} {trim_file} trim {start_time} {length};")
EXTRACT_FRAMES = ("ffmpeg -i {video_file} -vf fps={fps} {file} </dev/null > /dev/null 2>&1;")
FILE = ("{}.{}")

def url_video(youtube_id):
    return "https://www.youtube.com/watch?v=" + youtube_id

def extract_data(video_file, audio_path, frames_path, 
                 filename, audio_ext, sample_rate, fps):
    cmd = ""
    if audio_path != None:
        audio_file = os.path.join(audio_path, FILE.format(filename, audio_ext))
        cmd += EXTRACT_AUDIO.format(video_file=video_file,
                                    audio_ext=audio_ext,
                                    sample_rate=sample_rate,
                                    audio_file=audio_file)
    if frames_path != None:
        frame_files = os.path.join(frames_path, FILE.format(filename + ":%02d", "jpg"))
        cmd += EXTRACT_FRAMES.format(video_file=video_file,
                                     fps=fps,
                                     sample_rate=sample_rate,
                                     file=frame_files)
    os.system(cmd)
    os.remove(video_file)

def cut_audio(youtube_id, audio_path, filename, 
              start_time, end_time, ext="wav"):
    file = os.path.join(audio_path, FILE.format(filename, ext))
    trim_file= os.path.join(tempfile.gettempdir(), FILE.format("trim_" + filename, ext))
    length = end_time - start_time
    cmd = CUT_AUDIO.format(file=file,
                           trim_file=trim_file,
                           start_time=start_time, 
                           length=length)
    os.system(cmd)
    os.remove(file)
    shutil.move(trim_file, file)

def download_data(youtube_id, filename, start_time, 
                  end_time, audio_path, frames_path, fps=25,
                  video_ext="mp4", audio_ext="wav", sample_rate=16000):
    video_file = os.path.join(tempfile.gettempdir(), FILE.format(filename, video_ext))
    cmd = DL_VIDEO.format(url=url_video(youtube_id),
                          start_time=start_time,
                          end_time=end_time,
                          file=video_file)
    os.system(cmd)
    if os.path.exists(video_file):
        extract_data(video_file, audio_path, frames_path, filename, audio_ext, sample_rate, fps)
        #cut_audio(youtube_id, audio_path, filename, 0, 3)
        return True
    return False

