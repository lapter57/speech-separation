import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioStream(nn.Module):
    def __init__(self, config):
        super(AudioStream, self).__init__()
        self.config = config

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=96, kernel_size=(1, 7), padding=(0, 3), dilation=(1, 1)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(7, 1), padding=(3, 0), dilation=(1, 1)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), padding=(2, 2), dilation=(1, 1)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), padding=(4, 2), dilation=(2, 1)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), padding=(8, 2), dilation=(4, 1)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), padding=(16, 2), dilation=(8, 1)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), padding=(32, 2), dilation=(16, 1)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), padding=(64, 2), dilation=(32, 1)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), padding=(2, 2), dilation=(1, 1)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), padding=(4, 4), dilation=(2, 2)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), padding=(8, 8), dilation=(4, 4)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), padding=(16, 16), dilation=(8, 8)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), padding=(32, 32), dilation=(16, 16)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), padding=(64, 64), dilation=(32, 32)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=8, kernel_size=(1, 1), padding=(0, 0), dilation=(1, 1)),
            nn.BatchNorm2d(8), nn.ReLU()
        )
    
    def forward(self, x):
        # x: [B, F, T, 2]
        x = x.permute(0, 3, 2, 1)
        # x: [B, 2, T, F]
        return self.conv(x)
        # x: [B, 8, T, F]

class VisualStream(nn.Module):
    def __init__(self, config):
        super(VisualStream, self).__init__()
        self.config = config

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(7, 1), padding=(3, 0), dilation=(1, 1)),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 1), padding=(2, 0), dilation=(1, 1)),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 1), padding=(4, 0), dilation=(2, 1)),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 1), padding=(8, 0), dilation=(4, 1)),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 1), padding=(16, 0), dilation=(8, 1)),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 1), padding=(32, 0), dilation=(16, 1)),
            nn.BatchNorm2d(256), nn.ReLU()
        )

    def forward(self, x):
        # x: [B, num_frames, 1, emb_size, num_speakers]
        x = x.permute(0, 3, 1, 2, 4)
        # x: [B, emb_size, num_frames, 1, num_speakers]
        visual_list = list()
        for i in range(self.config.data.num_speakers):
            v = x[:,:,:,:,i]
            # v: [B, emb_size, num_frames, 1]
            v = self.conv(v)
            # v: [B, 256, num_frames, 1]
            v = v.permute(0, 3, 1, 2)
            # v: [B, 1, 256, num_frames]
            v = nn.functional.interpolate(v, size=(256, self.config.audio.num_freq), mode="bilinear")
            # v: [B, 1, 256, F]
            v = v.permute(0, 2, 3, 1)
            # v: [B, 256, F, 1]
            visual_list.append(v)
        x = torch.cat(visual_list, dim=1)
        # x: [B, 256 * num_speakers, F, 1]
        return x.view((-1, x.shape[1], x.shape[2]))
        # x: [B, 256 * num_speakers, F]

class FusionStream(nn.Module):
    def __init__(self, config, is_av=True):
        super(FusionStream, self).__init__()
        self.config = config

        lstm_input_size = 8 * config.audio.num_time
        lstm_input_size = lstm_input_size + 256 * config.data.num_speakers if is_av else lstm_input_size

        self.blstm = nn.LSTM(
            lstm_input_size,
            config.model.lstm_dim,
            batch_first=True,
            bidirectional=True)

        self.fc1 = nn.Linear(2 * config.model.lstm_dim, config.model.fc1_dim)
        self.fc2 = nn.Linear(config.model.fc1_dim, config.model.fc2_dim)
        self.fc3 = nn.Linear(config.model.fc2_dim, config.audio.num_time * 2 * config.data.num_speakers)

    def forward(self, x):
        x, _ = self.blstm(x)
        # x: [B, F, 2 * lstm_dim]
        x = F.relu(x)
        x = self.fc1(x) 
        # x: [B, F, fc2_dim]
        x = F.relu(x)
        x = self.fc2(x)
        # x: [B, F, fc3_dim]
        x = F.relu(x)
        x = self.fc3(x)
        # x: [B, F, 2 * T * num_speakers]
        x = torch.sigmoid(x)

        spec_size = 2 * self.config.audio.num_time
        masks = []
        for i in range(self.config.data.num_speakers):
            start = i * spec_size
            mask = x[:, :, start:start+spec_size]
            masks.append(mask)
        x = torch.stack(masks, dim=0)
        # x: [2, B, F, 2 * T]
        x = x.permute(1, 0, 2, 3) 
        # x: [B, 2, F, 2 * T]
        x = x.view(-1, self.config.data.num_speakers, self.config.audio.num_time, 2, self.config.audio.num_freq)
        # x: [B, 2, T, 2, F]
        x = x.permute(0, 1, 3, 2, 4)
        # x: [B, 2, 2, T, F]
        return  x.permute(0, 4, 3, 2, 1)
        # x: [B, F, T, 2, 2]
     

class AoModel(nn.Module):
    def __init__(self, config):
        super(AoModel, self).__init__()
        self.config = config
        
        self.audio_stream = AudioStream(config)
        self.fusion_stream = FusionStream(config, False)
        
    def forward(self, x):
        # x: [B, F, T, 2]
        x = self.audio_stream(x)
        # x: [B, 8, T, F]
        x = x.contiguous().view((-1, x.shape[3], x.shape[1] * x.shape[2]))
        # x: [B, F, 8 * T]
        return self.fusion_stream(x)
        # x: [B, F, T, 2, 2]
        

class AvModel(nn.Module):
    def __init__(self, config):
        super(AvModel, self).__init__()
        self.config = config

        self.audio_stream = AudioStream(config)
        self.visual_stream = VisualStream(config)
        self.fusion_stream = FusionStream(config)

    def forward(self, a, v):
        # a: [B, F, T, 2]
        # v: [B, num_frames, 1, emb_size, num_speakers]
        a = self.audio_stream(a)
        # a: [B, 8, T, F]
        a = a.contiguous().view((-1, a.shape[1] * a.shape[2], a.shape[3]))
        # a: [B, 8 * T, F]
        v = self.visual_stream(v)
        # v: [B, 256 * num_speakers, F]
        fusion = torch.cat([v, a], dim=1)
        # fusion: [B, 256 * num_speakers + 8 * T, F]
        fusion = fusion.permute(0, 2, 1)
        # fusion: [B, F, 256 * num_speakers + 8 * T]
        return self.fusion_stream(fusion)
        # fusion: [B, F, T, 2, 2]

