
import torch
import torch.nn as nn
from .blocks import DownBlock, MidBlock, UpBlock


class VQVAE(nn.Module):

    def __init__(self, im_channels=3, model_config=None):
        super().__init__()

        self.down_channels = [64, 128, 256, 256]
        self.mid_channels = [256, 256]
        self.down_sample = [True, True, True]
        self.num_down_layers = 2
        self.num_mid_layers = 2
        self.num_up_layers = 2
        
        # To disable attention in Downblock of Encoder and Upblock of Decoder
        self.attns = [False, False, False]
        
        # Latent Dimension
        self.z_channels = 3
        self.codebook_size = 8192
        self.norm_channels = 32
        self.num_heads = 4

        # Assertion to validate the channel information
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-1]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1

        # Wherever we use downsampling in encoder correspondingly use upsampling in decoder
        self.up_sample = list(reversed(self.down_sample))

        ##################### Encoder ######################
        self.encoder_conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))

        # Down + Mid
        self.encoder_downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.encoder_downs.append(
                DownBlock(self.down_channels[i], self.down_channels[i+1], t_emb_dim=None, 
                          num_heads=self.num_heads, num_layers=self.num_down_layers,
                          norm_channels=self.norm_channels, 
                          down_sample=self.down_sample[i],
                          attn=self.attns[i])
            )
        
        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mids.append(
                MidBlock(self.mid_channels[i], self.mid_channels[i+1], t_emb_dim=None,
                         num_heads=self.num_heads, num_layers=self.num_mid_layers,
                         norm_channels=self.norm_channels)
            )

        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = nn.Conv2d(self.down_channels[-1], self.z_channels, kernel_size=3, padding=1)

        # Pre-quantization Conv
        self.pre_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)

        # Codebook for VQVAE
        self.embedding = nn.Embedding(self.codebook_size, self.z_channels)
        ####################################################

        ##################### Decoder ######################
        self.post_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        self.decoder_conv_in = nn.Conv2d(self.z_channels, self.mid_channels[-1], kernel_size=3, padding=1)

        # Mid + Up
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(1, len(self.mid_channels))):
            self.decoder_mids.append(
                MidBlock(self.mid_channels[i], self.mid_channels[i-1], t_emb_dim=None,
                         num_heads=self.num_heads, num_layers=self.num_mid_layers,
                         norm_channels=self.norm_channels)
            )

        self.decoder_ups = nn.ModuleList([])
        for i in reversed(range(1, len(self.down_channels))):
            self.decoder_ups.append(
                UpBlock(self.down_channels[i], self.down_channels[i-1], t_emb_dim=None,
                        up_sample=self.down_sample[i-1],
                        num_heads=self.num_heads,
                        num_layers=self.num_up_layers,
                        attn=self.attns[i-1],
                        norm_channels=self.norm_channels)
            )

        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        self.decoder_conv_out = nn.Conv2d(self.down_channels[0], im_channels, kernel_size=3, padding=1)
        ####################################################

    def forward(self, x):
        z, quant_losses = self.encode(x)
        out = self.decode(z)
        return out, z, quant_losses
    
    def quantize(self, x):
        B, C, H, W = x.shape
        # -> B, H, W, C -> B, H*W, C
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)

        # Find nearest embedding/codebook vector
        # dist between (B, H*W, C) and (B, K, C) -> (B, H*W, K)
        dist = torch.cdist(x, self.embedding.weight[None, :].repeat((B, 1, 1)))
        min_encoding_indices = torch.argmin(dist, dim=-1)  # (B, H*W)

        # Replace encoder output with nearest codebook
        # quant_out -> B*H*W, C
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1))

        # x -> B*H*W, C
        x = x.reshape(-1, C)
        commmitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
        quantize_losses = {
            'codebook_loss': codebook_loss,
            'commitment_loss': commmitment_loss
        }
        # Straight through estimation
        quant_out = x + (quant_out - x).detach()

        # quant_out -> B, C, H, W
        quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
        min_encoding_indices = min_encoding_indices.reshape((-1, quant_out.size(-2), quant_out.size(-1)))
        return quant_out, quantize_losses, min_encoding_indices

    def encode(self, x):
        out = self.encoder_conv_in(x)
        for idx, down in enumerate(self.encoder_downs):
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)
        
        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        out = self.pre_quant_conv(out)
        out, quant_losses, _ = self.quantize(out)
        return out, quant_losses
    
    def decode(self, z):
        out = z
        out = self.post_quant_conv(out)
        out = self.decoder_conv_in(out)
        for mid in self.decoder_mids:
            out = mid(out)
        for idx, up in enumerate(self.decoder_ups):
            out = up(out)
        
        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        return out
