import os
import time
import logging
import math
import torch
import torch.nn as nn
import traceback

from mir_eval.separation import bss_eval_sources
from config import Config
from writer import CustomWriter
from generators import create_dataloader
from audio import Audio
from models import AoModel, AvModel


class Trainer():
    def __init__(self, config, chkpt_file=None):
        self.config = config
        self.pt_dir = os.path.join(config.log.chkpt_dir, config.log.model_name)
        self.log_dir = os.path.join(config.log.log_dir, config.log.model_name)
        os.makedirs(self.pt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        logging.basicConfig(
           level=logging.INFO,
           format='%(asctime)s - %(levelname)s - %(message)s',
           handlers=[
               logging.FileHandler(os.path.join(self.log_dir,
               '%s-%d.log' % (config.log.model_name, time.time()))),
               logging.StreamHandler()
           ]
        )

        self.logger = logging.getLogger()
        self.writer = CustomWriter(config, self.log_dir)

        self.trainloader = create_dataloader(config, True)
        self.testloader = create_dataloader(config, False)

        self.audio = Audio(config)
        self.model = AoModel(config).cuda() if config.train.model == "ao" else AvModel(config).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.train.adam_lr)
        self.criterion = nn.MSELoss()

        self.step = 0

        if chkpt_file is not None:
            self.logger.info("Resuming from checkpoint: %s" % chkpt_file)
            checkpoint = torch.load(chkpt_file)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.step = checkpoint['step']
            self.logger.info("Starting new training run")

    def predict(self, mix, embs):
        return self.model(mix) if embs is None else self.model(mix, embs.cuda())

    def get_estimated_specs(self, mix, masks_batch):
        sep_list = list()
        for i in range(self.config.data.num_speakers):
            mask = masks_batch[:,:,:,i]
            sep_list.append(mask * mix)
        return torch.stack(sep_list, dim=3)

    def validate(self):
        self.model.eval()
    
        with torch.no_grad():
            for mix, targets, embs in self.testloader:
                mix = mix.cuda()
                targets = targets.cuda()
    
                masks = self.predict(mix, embs) 
               # masks[masks == 0] += 0.000001
               # masks[masks == 1] -= 0.000001
               # masks = torch.log(masks / (1 - masks)) 
                masks_batch = masks[0]

                est = self.get_estimated_specs(mix[0], masks_batch)  
                est = est.unsqueeze(0) 
                
                test_loss = self.criterion(est.cuda(), targets).item()

                mixed = mix[0].cpu().detach().numpy()
                mixed_wav = self.audio.spec2wav(mixed)
                targets = targets[0].cpu().detach().numpy()
                targets_wavs = list()
                for i in range(self.config.data.num_speakers):
                    targets_wavs.append(self.audio.spec2wav(targets[:,:,:,i]))
                est = est[0].cpu().detach().numpy()
                est_wavs = list()
                for i in range(self.config.data.num_speakers):
                    est_wavs.append(self.audio.spec2wav(est[:,:,:,i]))
                est_masks = masks_batch.cpu().detach().numpy()

                sdrs = list()
                for i in range(self.config.data.num_speakers):
                    sdrs.append(bss_eval_sources(targets_wavs[i], est_wavs[i], False)[0][0])
                self.writer.log_evaluation(test_loss, sdrs,
                                           mixed_wav, targets_wavs, est_wavs,
                                           mixed[:,:,0], targets[:,:,0,:], est[:,:,0,:], est_masks[:,:,0,:],
                                           self.step)
                break

        self.model.train()

    def train(self):
        try:
            while True:
                self.model.train()
                for mix, targets, embs in self.trainloader:
                    mix = mix.cuda()
                    targets = targets.cuda()

                    masks = self.predict(mix, embs) 
                   # masks[masks == 0] += 0.000001
                   # masks[masks == 1] -= 0.000001
                   # masks = torch.log(masks / (1 - masks)) 
                    est_list = list()
                    for i in range(self.config.train.batch_size):
                        masks_batch = masks[i]
                        est_list.append(self.get_estimated_specs(mix[i], masks_batch))
                    est = torch.stack(est_list, dim=0)
                        
                    loss = self.criterion(est.cuda(), targets)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.step += 1

                    loss = loss.item()
                    if loss > 1e8 or math.isnan(loss):
                        self.logger.error("Loss exploded to %.02f at step %d!" % (loss, step))
                        raise Exception("Loss exploded")

                    if self.step % self.config.train.summary_interval == 0:
                        self.writer.log_training(loss, self.step)
                        self.logger.info("Wrote summary at step %d" % self.step)

                    if self.step % self.config.train.checkpoint_interval == 0:
                        save_path = os.path.join(self.pt_dir, 'chkpt_%d.pt' % self.step)
                        torch.save({
                            'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'step': self.step,
                        }, save_path)
                        self.logger.info("Saved checkpoint to: %s" % save_path)
                        self.validate()
        except Exception as e:
            self.logger.info("Exiting due to exception: %s" % e)
            traceback.print_exc()

