import torch
import soundfile as sf
from stft_and_isft import stft_multich, istft_batch, SR
from model import mvdr_beamform
from metric import snr_db, mse_db

def enhance_one(noisy_path, clean_path, model, device):
    model.eval()
    noisy, sr1 = sf.read(noisy_path, always_2d=True)   # (T,2)
    clean, sr2 = sf.read(clean_path)                   # (T,)
    assert sr1 == sr2 == SR

    noisy_t = torch.from_numpy(noisy.T).float().unsqueeze(0).to(device)  # (1,2,T)
    with torch.no_grad():
        Y = stft_multich(noisy_t)         # (1,2,F,Tf)
        Y_ref = Y[:, 0]
        M_s = model(Y_ref)
        Z   = mvdr_beamform(Y, M_s, ref_channel=0)
        enh_t = istft_batch(Z)           # (1,Te)
    enh = enh_t[0].cpu().numpy()

    L = min(len(clean), len(enh), noisy.shape[0])
    clean = clean[:L]
    enh   = enh[:L]
    noisy_mono = noisy[:L, 0]

    snr_in  = snr_db(clean, noisy_mono)
    snr_out = snr_db(clean, enh)
    osinr   = snr_out - snr_in
    mse_out = mse_db(clean, enh)

    return osinr, mse_out, clean, noisy_mono, enh

def enhance_and_save(noisy_path, out_path, model, device):
    noisy, sr = sf.read(noisy_path, always_2d=True)      # (T,2)
    noisy_t = torch.from_numpy(noisy.T).float().unsqueeze(0).to(device)  # (1,2,T)

    # DNN + MVDR enhancement
    with torch.no_grad():
        Y = stft_multich(noisy_t)        # (1,2,F,Tf)
        Y_ref = Y[:, 0]                  # reference channel
        M_s = model(Y_ref)               # speech mask
        Z   = mvdr_beamform(Y, M_s, ref_channel=0)  # beamformed STFT
        enh_t = istft_batch(Z)           # (1, Te)

    enh = enh_t[0].cpu().numpy()         # (Te,)

    # save as mono wav
    sf.write(out_path, enh, sr)
    return enh, sr
