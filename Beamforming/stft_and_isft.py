import torch

# STFT and iSTFT configuration
SR = 16000
N_FFT = 512
HOP = 128
WIN = torch.hann_window(N_FFT)

def stft_multich(x):
    """
    x: (B, C, T) real
    returns: (B, C, F, T_frames) complex
    """
    B, C, T = x.shape
    x2 = x.reshape(B * C, T)  # (B*C, T)

    Y = torch.stft(
        x2,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        window=WIN.to(x.device),
        return_complex=True,
    )  # (B*C, F, Tf)

    F_bins, Tf = Y.shape[-2], Y.shape[-1]
    Y = Y.view(B, C, F_bins, Tf)  # (B, C, F, Tf)
    return Y
def istft_batch(Z):
    """
    Z: (B, F, T_frames) complex
    returns: (B, T) real
    """
    outs = []
    for b in range(Z.shape[0]):
        zb = torch.istft(
            Z[b],
            n_fft=N_FFT,
            hop_length=HOP,
            win_length=N_FFT,
            window=WIN.to(Z.device),
        )
        outs.append(zb)
    maxL = max(z.shape[0] for z in outs)
    out = torch.zeros(len(outs), maxL, device=Z.device)
    for i, z in enumerate(outs):
        out[i, : z.shape[0]] = z
    return out
print("done")
