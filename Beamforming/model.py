# imports
import torch
import torch.nn as nn

from stft_and_isft import N_FFT

#CNN MASK ESTIMATOR
#import torch.nn as nn
# reference channel STFT, real/imag concatenation, 1x1 conv, residual dilated 3x3 stacks,
# final 1-channel mask with sigmoid
class ConvBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, C, F, T)
        y = self.conv(x)
        # layernorm over channel: permute to (B, F, T, C)
        y = y.permute(0,2,3,1)
        y = self.norm(y)
        y = y.permute(0,3,1,2)
        y = self.relu(y)
        return x + y  # residual
class MaskNet(nn.Module):
    def __init__(self, n_fft=N_FFT, n_stacks=2, n_layers=4, ch=16):
        super().__init__()
        self.in_conv = nn.Conv2d(2, ch, 1)   # 2: real+imag
        blocks = []
        for s in range(n_stacks):
            for l in range(n_layers):
                dilation = 2**l
                blocks.append(ConvBlock(ch, dilation))
        self.blocks = nn.Sequential(*blocks)
        self.out_conv = nn.Conv2d(ch, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, Y_ref):
        # Y_ref: (B, F, T) complex
        Yr = Y_ref.real
        Yi = Y_ref.imag
        x = torch.stack([Yr, Yi], dim=1)  # (B, 2, F, T)
        x = self.in_conv(x)
        x = self.blocks(x)
        x = self.out_conv(x)
        M = self.sigmoid(x)               # (B,1,F,T)
        return M.squeeze(1)               # (B,F,T)
#MASK BASED COVARIANCE AND MVDR
def compute_covariance(Y, M):
    """
    Y: (B, C, F, T) complex
    M: (B, F, T) real mask
    returns: (B, F, C, C) complex
    """
    B, C, F, T = Y.shape
    # apply mask over time frames
    M_ = M.unsqueeze(1)                 # (B,1,F,T)
    Yw = Y * M_                         # (B,C,F,T)
    # move time to last dim, channels before it
    Yw = Yw.permute(0, 2, 3, 1)         # (B,F,T,C)
    # covariance for each freq bin
    R_list = []
    for f in range(F):
        Yf = Yw[:, f]                   # (B,T,C)
        YfH = Yf.conj().transpose(1, 2) # (B,C,T)
        Rf = torch.matmul(YfH, Yf) / (Yf.shape[1] + 1e-8)  # (B,C,C)
        R_list.append(Rf)
    R = torch.stack(R_list, dim=1)      # (B,F,C,C)
    return R
def mvdr_beamform(Y, M_speech, ref_channel=0):
    B, C, F, T = Y.shape
    M_noise = 1.0 - M_speech

    R_s = compute_covariance(Y, M_speech)  # (B,F,C,C) complex
    R_n = compute_covariance(Y, M_noise)

    eps = 1e-6
    eye = torch.eye(C, device=Y.device, dtype=torch.complex64).view(1, 1, C, C)
    R_n = R_n.to(torch.complex64)
    R_s = R_s.to(torch.complex64)
    R_n_inv = torch.linalg.inv(R_n + eps * eye)

    # steering vector: one-hot for reference channel, cast to complex
    u = torch.zeros(C, 1, device=Y.device)
    u[ref_channel, 0] = 1.0
    u = u.to(torch.complex64)
    u = u.view(1, 1, C, 1).repeat(B, F, 1, 1)  # (B,F,C,1)

    w = torch.matmul(R_n_inv, u)              # (B,F,C,1) complex

    denom = torch.matmul(
        w.conj().transpose(2, 3), torch.matmul(R_s, w)
    )                                          # (B,F,1,1)
    w = w / (denom.sqrt() + eps)
    Yp = Y.to(torch.complex64).permute(0, 2, 3, 1)  # (B,F,T,C)
    Yp = Yp.unsqueeze(-1)                            # (B,F,T,C,1)
    wH = w.conj().permute(0, 1, 3, 2)               # (B,F,1,C)
    Z = torch.matmul(wH.unsqueeze(2), Yp)           # (B,F,T,1,1)
    Z = Z.squeeze(-1).squeeze(-1)                  # (B,F,T) complex
    return Z
