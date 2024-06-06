import math

import torch
import torch.nn as nn


class EncoderLayer(nn.Module):
    '''
    The transformer encoder layer
    '''
    def __init__(self, opt):
        super(EncoderLayer, self).__init__()
        
        self.mha = nn.MultiheadAttention(embed_dim=opt.fog_model_dim, 
                                         num_heads=opt.fog_model_num_heads, 
                                         dropout=opt.fog_model_mha_dropout)
        
        self.layernorm = nn.LayerNorm(opt.fog_model_dim)
        
        self.seq = nn.Sequential(
            nn.Linear(opt.fog_model_dim, opt.fog_model_dim),
            nn.ReLU(),
            nn.Dropout(opt.fog_model_encoder_dropout),
            nn.Linear(opt.fog_model_dim, opt.fog_model_dim),
            nn.Dropout(opt.fog_model_encoder_dropout)
        )
        
    def forward(self, x):
        # x: (B,S,D)
        attn_output, _ = self.mha(query=x, key=x, value=x)
        x = x + attn_output
        x = self.layernorm(x)
        x = x + self.seq(x)
        x = self.layernorm(x)
        return x  # (B,S,D)


class Encoder(nn.Module):
    '''
    Encoder is a combination of transformer encoder and two BidirectionalLSTM layers
    '''
    def __init__(self, opt):
        super(Encoder, self).__init__()
    
        self.first_linear = nn.Linear(opt.fog_model_input_dim, opt.fog_model_dim)
        
        self.first_dropout = nn.Dropout(opt.fog_model_first_dropout)
        
        self.enc_layers = nn.Sequential()
        for _ in range(opt.fog_model_num_encoder_layers):
            self.enc_layers.append(EncoderLayer(opt))
            
        self.lstm_layers = nn.LSTM(opt.fog_model_dim, opt.fog_model_dim, 
                                   num_layers=opt.fog_model_num_lstm_layers,
                                   batch_first=True, bidirectional=True)
        
        self.sequence_len = opt.block_size // opt.patch_size # 864
        self.pos_encoding = nn.Parameter(torch.randn(1, self.sequence_len,
                                                     opt.fog_model_dim) * 0.02, 
                                         requires_grad=True)
        
    def forward(self, x, training=True):
        # x: (B, S, P*num_feats), Example shape (32, 864, 54)
        batch_size = x.size(0)
        x = x / 25.0  # Normalization attempt in the segment [-1, 1]
        x = self.first_linear(x)  # (batch_size, sequence_len, fog_model_dim)

        if training:  # augmentation by randomly roll of the position encoding tensor
            shifts = torch.randint(low=-self.sequence_len, high=0, size=(batch_size,))
            random_pos_encoding = torch.cat([torch.roll(self.pos_encoding, 
                                                        shifts=s.item(), 
                                                        dims=1) \
                                             for s in shifts], dim=0)
            x = x + random_pos_encoding
        else:  # without augmentation
            x = x + self.pos_encoding.repeat(batch_size, 1, 1)
        
        x = self.first_dropout(x)
        
        x = self.enc_layers(x) # (B,S,D)
        
        x, _ = self.lstm_layers(x)

        return x  # (B,S,D*2), e.g. (32, 864, 640)
    
    
class TransformerBiLSTM(nn.Module):
    def __init__(self, opt):
        super(TransformerBiLSTM, self).__init__()
        
        self.encoder = Encoder(opt)
        self.last_linear = nn.Linear(opt.fog_model_dim * 2, 1)
        
    def forward(self, x):  
        # x: (B,S,P*num_feats), e.g. (32, 864, 54)
        x = self.encoder(x)  # (B,S,D*2), e.g. (32, 864, 640)
        x = self.last_linear(x)  # (B,S,1), e.g. (32, 864, 1)
        x = torch.sigmoid(x)  # Sigmoid activation
        return x  # (B,S,1)
    
    