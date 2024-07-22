import torch
import torch.nn as nn
import numpy as np
import time
import math


class PositionalEncoding(nn.Module):
    def __init__(self, feat_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, feat_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feat_dim, 2).float() * (-math.log(10000.0) / feat_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (BS, max_len, feat_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (BS, window, feat_dim)
        return x + self.pe[:, :x.size(1), :]
          

class Transformer(nn.Module):
    def __init__(self,
        input_dim=3,
        feat_dim=250,
        nheads=10, 
        nlayers=1,
        dropout=0.1
    ):
        super(Transformer, self).__init__()
        self.input_embedding  = nn.Linear(input_dim, feat_dim)
        # self.src_mask = None

        self.pos_encoder = PositionalEncoding(feat_dim=feat_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(
                                        d_model=feat_dim, 
                                        nhead=nheads, 
                                        dropout=dropout,
                                        batch_first=True)
        
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers=nlayers)
        
        self.decoder = nn.Linear(feat_dim,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)


    def forward(self,src):
        # src: (BS, window, 3)

        # if self.src_mask is None or self.src_mask.size(0) != len(src):
        #     device = src.device
        #     mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #     self.src_mask = mask

        x = self.input_embedding(src) # (BS, window, feat_dim)
        x = self.pos_encoder(x) # (BS, window, feat_dim)
        
        output = self.transformer_encoder(x) #, self.src_mask)
        output = self.decoder(output)
        output = torch.sigmoid(output)
        return output # (BS, window, 1)
