import tensorflow as tf
import numpy as np
import os

NOISE_PREFIX = "n:"

def stft(data, frame_length=400, frame_step=160, pad_end=False):
    data = power_law(data)
    stft = tf.signal.stft(data, frame_length, frame_step, pad_end=pad_end).numpy()
    return np.stack((stft.real, stft.imag), -1)

def istft(stft, frame_length=400, frame_step=160, with_pad=True):
    stft = stft[..., 0] + 1j * stft[..., 1]
    istft = tf.signal.inverse_stft(stft, frame_length, frame_step).numpy()
    if with_pad:
        padding = np.zeros((40,))
        istft = np.concatenate((padding, istft, padding), axis=0)
    print(istft.shape)
    return power_law(istft, inv=True)

def cRM(clean, mix, K=10, C=0.1):
    M = build_cRM(clean, mix)
    return compress_mask_with_tanh(M, K, C)

def power_law(data, power=0.3, inv=False):
    return np.sign(data) * (np.abs(data)) ** (power if inv else 1.0 / power)

def icRM(mix, cRM, K=10, C=0.1):
    M = recover_uncompressed_mask(cRM)
    clean = np.zeros_like(M)
    clean[..., 0] = M[..., 0] * mix[..., 0] - M[..., 1] * mix[..., 1]
    clean[..., 1] = M[..., 0] * mix[..., 1] + M[..., 1] * mix[..., 0]
    return clean
    
def build_cRM(clean, mix):
    M = np.zeros(mix.shape)
    numerator_real = mix[..., 0] * clean[..., 0] + mix[..., 1] * clean[..., 1]
    numerator_img = mix[..., 0] * clean[..., 1] - mix[..., 1] * clean[..., 0]
    denominator = mix[..., 0] ** 2 + mix[..., 1] ** 2
    M[..., 0] = numerator_real / denominator
    M[..., 1] = numerator_img / denominator
    return M

def compress_mask_with_tanh(M, K=10, C=0.1):
    numerator = 1 - np.exp(-C * M)
    denominator = 1 + np.exp(-C * M)
    return K * (numerator / denominator)

def recover_uncompressed_mask(M, K=10, C=0.1):
    numerator = K - M
    denominator = K + M
    return (-1 / C) * np.log(numerator / denominator)


def get_files(path):
    return np.array([os.path.join(path, f) for f in os.listdir(path) 
                     if os.path.isfile(os.path.join(path, f))])

def basename(path):
    return os.path.splitext(os.path.basename(path))[0]

def find_paths_contains(name, paths):
    result = np.array()
    for path in paths:
        if name in path:
            result.append(path)
    return result

def get_clean_in_mix(mix_filename):
    return np.array([s for s in mix_filename.split(".") 
                     if ":" in s and not s.startswith(NOISE_PREFIX)])
