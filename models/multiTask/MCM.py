# self supervised multimodal multi-task learning network
import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.subNets.BertTextEncoder import BertTextEncoder

__all__ = ['MCM']

class MCM(nn.Module):
    def __init__(self, args):
        super(MCM, self).__init__()
        # text subnets
        self.aligned = args.need_data_aligned
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_finetune)

        # audio-vision subnets
        audio_in, video_in = args.feature_dims[1:]
        self.audio_model = AuViSubNet(audio_in, args.a_lstm_hidden_size, args.audio_out, \
                            num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        self.video_model = AuViSubNet(video_in, args.v_lstm_hidden_size, args.video_out, \
                            num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)

        # the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(args.text_out + args.video_out + args.audio_out, args.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(args.post_fusion_dim, args.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(args.post_fusion_dim, 1)

        # the classify layer for text
        self.post_text_dropout = nn.Dropout(p=args.post_text_dropout)
        self.post_text_layer_1 = nn.Linear(args.text_out, args.post_text_dim)
        self.post_text_layer_2 = nn.Linear(args.post_text_dim, args.post_text_dim)
        self.post_text_layer_3 = nn.Linear(args.post_text_dim, 1)

        # the classify layer for audio
        self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)
        self.post_audio_layer_1 = nn.Linear(args.audio_out, args.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(args.post_audio_dim, args.post_audio_dim)
        self.post_audio_layer_3 = nn.Linear(args.post_audio_dim, 1)

        # the classify layer for video
        self.post_video_dropout = nn.Dropout(p=args.post_video_dropout)
        self.post_video_layer_1 = nn.Linear(args.video_out, args.post_video_dim)
        self.post_video_layer_2 = nn.Linear(args.post_video_dim, args.post_video_dim)
        self.post_video_layer_3 = nn.Linear(args.post_video_dim, 1)

        # the classify layer for text/video
        self.post_tv_dropout = nn.Dropout(p=0.1)
        self.post_tv_layer_1 = nn.Linear(512, 128)
        self.post_tv_layer_2 = nn.Linear(128, 128)
        self.post_tv_layer_3 = nn.Linear(128, 1)

        # the classify layer for text/audio
        self.post_ta_dropout = nn.Dropout(p=0.1)
        self.post_ta_layer_1 = nn.Linear(512, 128)
        self.post_ta_layer_2 = nn.Linear(128, 128)
        self.post_ta_layer_3 = nn.Linear(128, 1)

        # the classify layer for audio/video
        self.post_va_dropout = nn.Dropout(p=0.1)
        self.post_va_layer_1 = nn.Linear(512, 128)
        self.post_va_layer_2 = nn.Linear(128, 128)
        self.post_va_layer_3 = nn.Linear(128, 1)

        # Gate Mechanism
        t_dim = 768
        a_dim = args.audio_out
        v_dim = args.video_out
        self.ta_gating = BiModal_Gating(t_dim, a_dim, 512)
        self.tv_gating = BiModal_Gating(t_dim, v_dim, 512)
        self.va_gating = BiModal_Gating(v_dim, a_dim, 512)

        self.g_dropout = nn.Dropout(p=0.6)
        self.g2_dropout = nn.Dropout(p=0)
        self.g_layer = nn.Linear(512 * 3 + t_dim + a_dim + v_dim, 128)
        self.g_layer2 = nn.Linear(128, 128)
        self.g_predict = nn.Linear(128, 1)

    def forward(self, text, audio, video):
        audio, audio_lengths = audio
        video, video_lengths = video

        mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze().int().detach().cpu()
        text = self.text_model(text)[:,0,:]

        if self.aligned:
            audio = self.audio_model(audio, text_lengths)
            video = self.video_model(video, text_lengths)
        else:
            audio = self.audio_model(audio, audio_lengths)
            video = self.video_model(video, video_lengths)
        
        # fusion
        fusion_h = torch.cat([text, audio, video], dim=-1)
        fusion_h = self.post_fusion_dropout(fusion_h)
        fusion_h = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)
        # # text
        text_h = self.post_text_dropout(text)
        text_h = F.relu(self.post_text_layer_1(text_h), inplace=False)
        # audio
        audio_h = self.post_audio_dropout(audio)
        audio_h = F.relu(self.post_audio_layer_1(audio_h), inplace=False)
        # vision
        video_h = self.post_video_dropout(video)
        video_h = F.relu(self.post_video_layer_1(video_h), inplace=False)

        # classifier-fusion
        x_f = F.relu(self.post_fusion_layer_2(fusion_h), inplace=False)
        output_fusion = self.post_fusion_layer_3(x_f)

        # classifier-text
        x_t = F.relu(self.post_text_layer_2(text_h), inplace=False)
        output_text = self.post_text_layer_3(x_t)

        # classifier-audio
        x_a = F.relu(self.post_audio_layer_2(audio_h), inplace=False)
        output_audio = self.post_audio_layer_3(x_a)

        # classifier-vision
        x_v = F.relu(self.post_video_layer_2(video_h), inplace=False)
        output_video = self.post_video_layer_3(x_v)

        # Gate Mechanism
        ta_fusion_h = self.ta_gating(text,audio)
        tv_fusion_h = self.tv_gating(text,video)
        va_fusion_h = self.va_gating(video,audio)

        ta_output = self.post_ta_dropout(ta_fusion_h)
        ta_output = F.relu(self.post_ta_layer_1(ta_output), inplace=False)
        ta_output = F.relu(self.post_ta_layer_2(ta_output), inplace=False)
        ta_output = self.post_ta_layer_3(ta_output)

        tv_output = self.post_tv_dropout(tv_fusion_h)
        tv_output = F.relu(self.post_tv_layer_1(tv_output), inplace=False)
        tv_output = F.relu(self.post_tv_layer_2(tv_output), inplace=False)
        tv_output = self.post_tv_layer_3(tv_output)

        va_output = self.post_va_dropout(va_fusion_h)
        va_output = F.relu(self.post_va_layer_1(va_output), inplace=False)
        va_output = F.relu(self.post_va_layer_2(va_output), inplace=False)
        va_output = self.post_va_layer_3(va_output)

        g_h = torch.cat([text, audio, video, ta_fusion_h, tv_fusion_h, va_fusion_h], dim=1)
        g_h1 = self.g_dropout(g_h)
        g_h2 = self.g2_dropout(g_h)

        g_h1 = F.relu(self.g_layer(g_h1), inplace=False)
        g_h1 = F.relu(self.g_layer2(g_h1), inplace=False)
        g_h1 = self.g_predict(g_h1)

        g_h2 = F.relu(self.g_layer(g_h2), inplace=False)
        g_h2 = F.relu(self.g_layer2(g_h2), inplace=False)
        g_h2 = self.g_predict(g_h2)

        res = {
            'M': g_h1,
            'M2': g_h2, 
            'T': output_text,
            'A': output_audio,
            'V': output_video,
            'TA': ta_output,
            'TV': tv_output,
            'VA': va_output
        }
        return res

class BiModal_Gating(nn.Module):
    def __init__(self, x_input_size, y_input_size, hidden_size):
        super().__init__()
        self.Linear1 = nn.Linear(x_input_size, hidden_size)
        self.Linear2 = nn.Linear(y_input_size, hidden_size)
        self.Linear3 = nn.Linear(x_input_size+y_input_size, hidden_size)

    def forward(self, x_input, y_input):
        h_1 = torch.tanh(self.Linear1(x_input))
        h_2 = torch.tanh(self.Linear2(y_input))
        f_input = torch.cat([x_input,y_input], dim=1)
        z_1 = F.relu(self.Linear3(f_input))
        output = z_1 * h_1 + (1 - z_1) * h_2
        return output

class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(AuViSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1
