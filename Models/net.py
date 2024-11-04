
import torch.nn as nn
import torch.nn.functional as F
import torch
from Models.utils_tf import *

FEATURE_DIM = 128

def re_4d(x):
    x = x.reshape(x.shape[0], x.shape[1]* x.shape[2], x.shape[3], x.shape[4])
    return x

def re_5d(x):
    return x.unsqueeze(1)

def main_conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3, stride=1,padding=1,bias=False),
        nn.BatchNorm3d(out_channel),
    )
    return layer
#####change
def main_conv3x3_0(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,bias=False),
        nn.BatchNorm2d(out_channel),
    )
    return layer

def main_conv3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3, stride=1,padding=1,bias=False),
        nn.BatchNorm2d(out_channel),
    )
    return layer

class ResMapping(nn.Module):
    def __init__(self, in_channels):
        super(ResMapping, self).__init__()
        self.conv1 = main_conv3x3x3(in_channels, in_channels)
        self.conv2 = main_conv3x3x3(in_channels, in_channels)

    def forward(self, x): 
        out = F.relu(self.conv1(x.unsqueeze(1)), inplace=True) 
        out = self.conv2(out)
        return x + out[:,0,:,:,:]

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = MS_MSA(dim, heads, dim_heads)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        
        # after_norm
        x = self.norm1(self.attn(x) + x)
        y = x
        x = self.ffn(x)
        out = self.norm2(y + x)
        
        # pre_norm
        # x = x + self.attn(self.norm1(x))
        # x = x + self.ffn(self.norm2(x))

        return out

class Main(nn.Module):
    def __init__(self, in_channel, out_channel1):
        super(Main, self).__init__()

        self.conv2d0 = main_conv3x3_0(100, FEATURE_DIM)
        self.resmapping= ResMapping(1)
        self.conv0 = main_conv3x3x3(in_channel, out_channel1)
        self.conv1 = main_conv3x3x3(in_channel, out_channel1)
        self.conv2 = main_conv3x3x3(in_channel, out_channel1)
        self.conv3 = main_conv3x3(FEATURE_DIM, FEATURE_DIM)
        self.TransformerBlock_H2 = TransformerBlock(128,4,32,1,False,'BiasFree_LayerNormb')
        self.TransformerBlock_L2 = TransformerBlock(128,4,32,1,False,'BiasFree_LayerNormb')
        self.TransformerBlock_H3 = TransformerBlock(128,4,32,1,False,'BiasFree_LayerNormb')
        self.TransformerBlock_L3 = TransformerBlock(128,4,32,1,False,'BiasFree_LayerNormb')

        self.TransformerBlock_H2_1 = TransformerBlock(128,4,32,1,False,'BiasFree_LayerNormb')
        self.TransformerBlock_L2_1 = TransformerBlock(128,4,32,1,False,'BiasFree_LayerNormb')
        self.TransformerBlock_H3_1 = TransformerBlock(128,4,32,1,False,'BiasFree_LayerNormb')
        self.TransformerBlock_L3_1 = TransformerBlock(128,4,32,1,False,'BiasFree_LayerNormb')

    def forward(self, x_in): # x_in: b*c*9*9
        x_in = F.relu(self.conv2d0(x_in))
        x = self.resmapping(x_in)
    
        #conv3d+downsampled
        x_L1_TMP = F.avg_pool3d(re_5d(x), kernel_size=(1,2,2), stride=(1,2,2))
        x_L1 = F.relu(self.conv0(x_L1_TMP),inplace=True) #b,c=1,d,h,w
        #X_H2 X_L2 -transformer
        x_H2_TMP = self.TransformerBlock_H2(x)
        x_H2_TMP = self.TransformerBlock_H2_1(x_H2_TMP)
        x_L2 = self.TransformerBlock_L2(re_4d(x_L1))
        x_L2 = self.TransformerBlock_L2_1(x_L2)
        #conv3d+upsampled
        x_L2_H2_TMP = F.interpolate(x_L2,size=(2*x_L2.shape[2]+1,2*x_L2.shape[2]+1),mode='bilinear',align_corners=True)
        x_L2_H2 = F.relu(self.conv1(re_5d(x_L2_H2_TMP)),inplace=True)
        x_H2 = torch.add(re_5d(x_H2_TMP), x_L2_H2)
        #X_H3 X_L3 -transformer
        x_H3_TMP = self.TransformerBlock_H3(re_4d(x_H2))
        x_H3_TMP = self.TransformerBlock_H3_1(x_H3_TMP)
        x_L3 = self.TransformerBlock_L3(x_L2)
        x_L3 = self.TransformerBlock_L3_1(x_L3)
        #conv3d+upsampled
        x_L3_H3_TMP = F.interpolate(x_L3, size=(2*x_L3.shape[2]+1,2*x_L3.shape[2]+1), mode='bilinear', align_corners=True)
        x_L3_H3 = F.relu(self.conv2(re_5d(x_L3_H3_TMP)),inplace=True)

        x_H3 = torch.add(re_5d(x_H3_TMP), x_L3_H3)

        Z = F.relu(self.conv3(re_4d(x_H3)), inplace=True)
        flatten = F.adaptive_avg_pool2d(Z,(1,1))
        embedding_feature = flatten.view(flatten.shape[0],-1)
        
        return embedding_feature

class VAE(nn.Module):
    def __init__(self, in_channel: int, latent_dim: int, hidden_dim: int):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
          nn.Linear(in_channel, hidden_dim),
          nn.BatchNorm1d(hidden_dim),
          nn.LeakyReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.Sequential(
          nn.Linear(hidden_dim, in_channel),
          nn.BatchNorm1d(in_channel),
          nn.Sigmoid()
        )
        
    def encode(self, input):
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]
    
    def decode(self, z):
        z = self.decoder_input(z)
        z_out = self.decoder(z)
        return z_out
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu, std + mu
    
    def forward(self, input):
        mu, log_var = self.encode(input) 
        z, z_inv = self.reparameterize(mu, log_var)
        z_out = self.decode(z)
        return [z_out, z_inv, input, mu, log_var]
    
    def loss_funcation(self, input, rec, mu, log_var, kld_weight = 0.00025):
        recons_loss = F.mse_loss(rec, input)
        kld_loss = torch.mean(-0.5* torch.sum(1+log_var-mu**2-log_var.exp(),dim=1),dim = 0)

        loss = recons_loss + kld_weight*kld_loss
        return loss

class Class_Adap(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Class_Adap, self).__init__()
        self.channel = channel
        self.fc = nn.Sequential(
            nn.Linear(channel, 64),
            nn.PReLU(),
            nn.Linear(64, 16),
            nn.PReLU(),
            nn.Linear(16, 1),
            nn.Softmax(dim = 0)
        )
    def forward(self, x):
        y = self.fc(x)
        
        return y
    
class Network(nn.Module):
    def __init__(self, input_dim = FEATURE_DIM, letent_dim = FEATURE_DIM, hidden_dim = FEATURE_DIM, adap_dim = 257):
        super(Network, self).__init__()

        self.vae = VAE(input_dim, letent_dim, hidden_dim)
        self.feature_enc = Main(1,1)
        self.Class_Adap = Class_Adap(adap_dim)

    def forward(self, x, domain = 'source'):
        
        feature = self.feature_enc(x)
    
        return feature,feature


