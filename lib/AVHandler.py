import os
import librosa
import scipy.io.wavfile as wavfile
import numpy as np
import random

DL_AUDIO = ("youtube-dl -x --audio-format {ext} -o {file} {url};")
CHANGE_SAMPLE_RATE = ("ffmpeg -i {old_file} -ar {sample_rate} -ac 1 {new_file};")
CUT_AUDIO = ("sox {file} {trim_file} trim {start_time} {length};")
FILE = ("{}.{}")

def url_video(youtube_id):
    return "https://www.youtube.com/watch?v=" + youtube_id

def download_audio(youtube_id, filename, path="", ext="wav", sample_rate=16000):
    tmp_file = os.path.join(path, FILE.format("temp_" + filename, ext))
    file = os.path.join(path, FILE.format(filename, ext))
    cmd = DL_AUDIO.format(ext=ext,
                          file=tmp_file, 
                          url=url_video(youtube_id))
    cmd += CHANGE_SAMPLE_RATE.format(old_file=tmp_file, 
                                     new_file=file, 
                                     sample_rate=sample_rate)
    os.system(cmd)
    try:
        os.remove(tmp_file)
        return True
    except OSError:
        return False
    
def cut_audio(youtube_id, start_time, end_time, filename, path="", ext="wav", with_remove=False):
    file = os.path.join(path, FILE.format(filename, ext))
    trim_file= os.path.join(path, FILE.format("trim_" + filename, ext))
    length = end_time - start_time
    cmd = CUT_AUDIO.format(file=file,
                           trim_file=trim_file,
                           start_time=start_time, 
                           length=length)
    os.system(cmd)
    if with_remove:
        try:
            os.remove(file)
            os.rename(trim_file, file)
        except OSError:
            pass
    
def mix_audio(paths, filename, noise_path=None):
    num_speakers = len(paths)
    mix = None
    sr = None
    for i in range(num_speakers):
        audio, audio_sr = librosa.load(paths[i], sr=None)
        audio = audio / np.max(audio)
        if i == 0:
            mix = audio
            sr = audio_sr
        else:
            mix += audio
    if noise_path != None:
        noise_files = [f for f in os.listdir(noise_path) if os.path.isfile(os.path.join(noise_path, f))]
        noise_file = random.choice(noise_files)
        noise, noise_sr = librosa.load(os.path.join(noise_path, noise_file), sr=None)
        noise = noise / np.max(noise)
        mix += 0.3 * noise
    wavfile.write(filename, sr, mix)
        