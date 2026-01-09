import os
import numpy as np
import torch

from infer import enhance_and_save, enhance_one
from data import test_pairs
from model import MaskNet
from metric import snr_db, mse_db, stoi_score, pesq_score
from stft_and_isft import SR
# e.g. first 10 test pairs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = MaskNet().to(device)

ckpt_path = "checkpoints/model.pt"
if os.path.exists(ckpt_path):
    net.load_state_dict(torch.load(ckpt_path, map_location=device))

subset = test_pairs[:10]

osinrs, mses = [], []
for noisy_path, clean_path in subset:
    os, md, _, _, _ = enhance_one(noisy_path, clean_path, net, device)
    osinrs.append(os)
    mses.append(md)

print("Mean OSINR over subset (dB):", float(np.mean(osinrs)))
print("Mean MSE (dB) over subset  :", float(np.mean(mses)))

noisy_path, clean_path = test_pairs[0]
out_path = "enhanced_test0.wav"

enhanced, sr = enhance_and_save(noisy_path, out_path, net.eval(), device)
print("Saved enhanced audio to:", out_path)

print(os.path.exists(out_path))

osinrs, mses, snrs_out, stois, pesqs = [], [], [], [], []
for noisy_path, clean_path in test_pairs:
    os, md, clean, noisy_mono, enh = enhance_one(noisy_path, clean_path, net, device)
    osinrs.append(os)
    mses.append(md)
    snrs_out.append(snr_db(clean, enh))
    stois.append(stoi_score(clean, enh, SR))
    pesqs.append(pesq_score(clean, enh, SR))

print("Test OSINR (dB):", float(np.mean(osinrs)))
print("Test MSE (dB):", float(np.mean(mses)))
print("Test SNR (dB):", float(np.mean(snrs_out)))
if any(v is not None for v in stois):
    print("Test STOI:", float(np.nanmean([v for v in stois if v is not None])))
else:
    print("Test STOI: unavailable (install pystoi)")
if any(v is not None for v in pesqs):
    print("Test PESQ:", float(np.nanmean([v for v in pesqs if v is not None])))
else:
    print("Test PESQ: unavailable (install pesq)")
