import numpy as np

from tensorboardX import SummaryWriter
from plotting import plot_spectrogram_to_numpy


class CustomWriter(SummaryWriter):
    def __init__(self, config, logdir):
        super(CustomWriter, self).__init__(logdir)
        self.config = config

    def log_training(self, train_loss, step):
        self.add_scalar('train_loss', train_loss, step)

    def log_evaluation(self, test_loss, sdr,
                       mixed_wav, targets_wavs, est_wavs,
                       mixed_spec, target_specs, est_specs, est_masks,
                       step):
        self.add_scalar('test_loss', test_loss, step)
        self.add_scalar('SDR', sdr, step)

        self.add_audio('mixed_wav', mixed_wav, step, self.config.audio.sample_rate)
        self.add_image('data/mixed_spectrogram',
            plot_spectrogram_to_numpy(mixed_spec), step, dataformats='HWC')
        for i in range(len(targets_wavs)):
            self.add_audio(('target_wav_{}').format(i), targets_wavs[i], step, self.config.audio.sample_rate)
            self.add_audio(('estimated_wav_{}').format(i), est_wavs[i], step, self.config.audio.sample_rate)
            self.add_image(('data/target_spectrogram{}').format(i),
                    plot_spectrogram_to_numpy(target_specs[:,:,i]), step, dataformats='HWC')
            self.add_image(('result/estimated_spectrogram{}').format(i),
                    plot_spectrogram_to_numpy(est_specs[:,:,i]), step, dataformats='HWC')
            self.add_image(('result/estimated_mask{}').format(i),
                    plot_spectrogram_to_numpy(est_masks[:,:,i]), step, dataformats='HWC')
            self.add_image(('result/estimation_error_sq{}').format(i),
                    plot_spectrogram_to_numpy(np.square(est_specs[:,:,i] - target_specs[:,:,i])), step, dataformats='HWC')

        
