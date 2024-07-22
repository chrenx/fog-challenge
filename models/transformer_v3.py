import math

import clip, torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, feat_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()    
        self.dropout = nn.Dropout(p=dropout)   
        pe = torch.zeros(max_len, feat_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feat_dim, 2).float() * (-math.log(10000.0) / feat_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (BS, max_len, feat_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (BS, window, feat_dim)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
    
class Transformer(nn.Module):
    def __init__(self,
        input_dim=7*3,
        feat_dim=250,
        nheads=10, 
        nlayers=1,
        dropout=0.1,
        clip_dim=512,
        feats=['lowerback_acc', 'l_midlatthigh_acc', 'l_latshank_acc', 'r_latshank_acc',
               'l_latshank_gyr', 'r_latshank_gyr', 'l_ankle_acc'],
        txt_cond=True,
        clip_version='ViT-B/32',
        activation='gelu'
    ):
        super(Transformer, self).__init__()

        self.feats = feats

        self.body_linear = nn.Linear(input_dim, feat_dim)
            
        self.pos_encoder = PositionalEncoding(feat_dim=feat_dim, dropout=dropout)

        self.encoder_layer = nn.TransformerEncoderLayer(
                                        d_model=feat_dim, 
                                        nhead=nheads, 
                                        dropout=dropout,
                                        batch_first=True,
                                        activation=activation)
        
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers=nlayers)
        
        self.decoder = nn.Linear(feat_dim,1)
        
        self.txt_cond = txt_cond
        if self.txt_cond:
            self.embed_text = nn.Linear(clip_dim, feat_dim)
            self.embed_txt_relu = nn.ReLU()
            self.clip_version = clip_version
            self.clip_model = self._load_and_freeze_clip(clip_version)
        
        self._init_weights()

    def _init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(clip_model)
        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model
    
    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 50
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            # [bs, context_length] # if n_tokens > context_length -> will truncate
            texts = clip.tokenize(raw_text, context_length=context_length, 
                                  truncate=True).to(device)
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], 
                                   dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            # [bs, context_length] # if n_tokens > 77 -> will truncate
            texts = clip.tokenize(raw_text, truncate=True).to(device)
        return self.clip_model.encode_text(texts).float()

    def forward(self,src):
        """
        src: {'lowerback_acc': (BS, window, 3), ... , 
              'event': }
        """
        
        # print(src.keys())
        # print(src['lowerback_acc'].shape)
        # exit(0)
        x = []
        for body_name in self.feats:
            if body_name == 'event':
                continue
            # (BS, window, 1, 3)
            x.append(src[body_name].unsqueeze(dim=-1).permute(0,1,3,2)) 
        x = torch.cat(x, dim=2)  # (BS, window, num_feats, 3)
        bs, window, n_feats, n_orient = x.shape
        x = x.reshape(bs, window, n_feats * n_orient) # (BS, window, 21)
        x = self.body_linear(x) # (BS, window, feat_dim)
        
        
        if self.txt_cond and 'event' in src.keys():
            # (window, bs)
            encode_txt = self.encode_text(src['event']) # (bs, clip_dim)
            # (bs, 1, feat_dim)
            emb_txt = self.embed_txt_relu(self.embed_text(encode_txt).unsqueeze(1))
            x = torch.cat([emb_txt, x], dim=1) # (bs, window+1, feat_dim)

        x = self.pos_encoder(x) # (BS, window, feat_dim)
        
        output = self.transformer_encoder(x)[:,1:,:] #, self.src_mask)
        output = self.decoder(output)
        output = torch.sigmoid(output)
        return output # (BS, window, 1)
