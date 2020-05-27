import librosa
import numpy as np
from scipy.special import expit, logit

EPSILON = 1e-8

class Audio():
    def __init__(self, config):
        self.config = config.audio

    def wav2spec(self, y):
        D = self.stft(y)
        S = self.amp_to_db(np.abs(D)) - self.config.ref_level_db
        S, D = self.normalize(S), np.angle(D)
        O = np.zeros((self.config.num_freq, self.config.num_time, 2))
        O[:,:,0] = S
        O[:,:,1] = D
        return O
   
    def spec2wav(self, spec):
        s, p = spec[:,:,0], spec[:,:,1]
        s = self.db_to_amp(self.denormalize(s) + self.config.ref_level_db)
        return self.istft(s, p)
 
    def amp_to_db(self, x):
        return 20.0 * np.log10(np.maximum(1e-5, x))
    
    def db_to_amp(self, x):
        return np.power(10.0, x * 0.05)
    
    def normalize(self, S):
        return np.clip(S / -self.config.min_level_db, -1.0, 0.0) + 1.0
    
    def denormalize(self, S):
        return (np.clip(S, 0.0, 1.0) - 1.0) * (-self.config.min_level_db)
    
    def stft(self, y):
        return librosa.stft(y=y, n_fft=self.config.n_fft,
                            hop_length=self.config.hop_length,
                            win_length=self.config.win_length)
    
    def istft(self, s, p):
        stft = s * np.exp(1j * p)
        return librosa.istft(stft,
                             hop_length=self.config.hop_length,
                             win_length=self.config.win_length)

    def power_law(self, data, inv=False):
        return np.sign(data) * (np.abs(data)) ** (self.config.audio.power if inv else 1.0 / self.config.audio.power)

    def crm(self, clean, mix):
        M = self.build_crm(clean, mix)
        return self.compress_mask(M)
    
    def icrm(self, mix, crm):
        M = self.recover_mask(crm)
        clean = np.zeros_like(M)
        clean[:,:,0] = M[:,:,0] * mix[:,:,0] - M[:,:,1] * mix[:,:,1]
        clean[:,:,1] = M[:,:,0] * mix[:,:,1] + M[:,:,1] * mix[:,:,0]
        return clean
        
    def build_crm(self, clean, mix):
        M = np.zeros(mix.shape)
        numerator_real = mix[:,:,0] * clean[:,:,0] + mix[:,:,1] * clean[:,:,1]
        numerator_img = mix[:,:,0] * clean[:,:,1] - mix[:,:,1] * clean[:,:,0]
        denominator = mix[:,:,0] ** 2 + mix[:,:,1] ** 2 + EPSILON
        M[:,:,0] = numerator_real / denominator
        M[:,:,1] = numerator_img / denominator
        return M
    
    def tanh_compress(self, M):
        K = self.config.crm.tanh_k
        C = self.config.crm.tanh_c
        numerator = 1 - np.exp(-C * M)
        numerator[numerator == np.inf] = 1
        numerator[numerator == -np.inf] = -1
        denominator = 1 + np.exp(-C * M)
        denominator[denominator == np.inf] = 1
        denominator[denominator == -np.inf] = -1
        return K * (numerator / denominator)

    def compress_mask(self, M):
        if self.config.crm.comressing == 'sigmoid':
            return expit(M)
        elif self.config.crm.compressing == 'tanh':
            return self.tanh_compress(M, K, C)
        return np.array([])
    
    def recover_mask(self, M):
        K = self.config.crm.tanh_k
        C = self.config.crm.tanh_c
        if self.config.crm.compressing == 'sigmoid':
            M = np.where(M == 0, EPSILON, M)
            M = np.where(M == 1, 1-EPSILON, M)
            return logit(M)
        elif self.config.crm.compressing == 'tanh':
            numerator = K - M
            denominator = K + M
            return (-1 / C) * np.log(numerator / denominator)
        return np.array([])

