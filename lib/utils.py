import tensorflow as tf

def power_law(data, power=0.3, inv=False):
    return np.sign(data) * (np.abs(data)) ** (power if inv else 1.0 / power)

def stft(data, frame_length=400, frame_step=160, pad_end=False):
    data = power_law(data)
    stft = tf.signal.stft(data, frame_length, frame_step, pad_end=pad_end).numpy()
    return np.stack((stft.real, stft.imag), -1)

def istft(stft, frame_length=400, frame_step=160, with_pad=True):
    stft = stft[...,0] + 1j * stft[...,1]
    istft = tf.signal.inverse_stft(stft, frame_length, frame_step).numpy()
    if with_pad:
        padding = np.zeros((40,))
        istft = np.concatenate((padding, istft, padding), axis=0)
    print(istft.shape)
    return power_law(istft, inv=True)