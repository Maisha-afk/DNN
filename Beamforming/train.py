import torch
import torch.optim as TO
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import numpy as np
from tqdm import tqdm

from data import train_pairs, val_pairs
from stft_and_isft import stft_multich, istft_batch, SR
from model import MaskNet, mvdr_beamform

MAX_LEN_SEC = 2.0
MAX_LEN = int(MAX_LEN_SEC * SR)

#ROOT = ""

def si_sdr_loss(est, ref, eps=1e-8):
    ref = ref - ref.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)
    proj = torch.sum(est * ref, dim=-1, keepdim=True) * ref
    proj = proj / (torch.sum(ref ** 2, dim=-1, keepdim=True) + eps)
    noise = est - proj
    ratio = torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps)
    return -10.0 * torch.log10(ratio + eps).mean()

class PairDataset(Dataset):
    def __init__(self, pairs, max_len_sec=4.0):
        self.pairs = pairs
        self.max_len = int(max_len_sec * SR)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]
        noisy, sr1 = sf.read(noisy_path, always_2d=True)  # (T,2)
        clean, sr2 = sf.read(clean_path)                  # (T,)
        assert sr1 == sr2 == SR

        # crop or pad both noisy and clean to exactly self.max_len
        T = min(len(clean), noisy.shape[0], self.max_len)

        noisy = noisy[:T]        # (T,2)
        clean = clean[:T]        # (T,)
        # pad at the end if shorter than max_len
        if T < self.max_len:
            pad_len = self.max_len - T
            noisy = np.pad(noisy, ((0, pad_len), (0, 0)), mode="constant")
            clean = np.pad(clean, (0, pad_len), mode="constant")

        noisy = noisy.T          # (2, max_len)
        return torch.from_numpy(noisy).float(), torch.from_numpy(clean).float()

train_ds = PairDataset(train_pairs, max_len_sec=MAX_LEN_SEC)
val_ds = PairDataset(val_pairs, max_len_sec=MAX_LEN_SEC)

train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, drop_last=True)
val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, drop_last=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = MaskNet().to(device)
opt = TO.Adam(net.parameters(), lr=1e-3)

for epoch in range(10):
    net.train()
    running = 0.0
    for noisy, clean in tqdm(train_dl):
        noisy = noisy.to(device)  # (B,2,T_fixed)
        clean = clean.to(device)  # (B,T_fixed)

        Y = stft_multich(noisy)   # (B,2,F,Tf)
        Y_ref = Y[:, 0]
        M_s = net(Y_ref)
        Z   = mvdr_beamform(Y, M_s, ref_channel=0)
        enh = istft_batch(Z)      # (B,Te)

        # crop to min length between enh and clean (should be very close)
        L = min(enh.shape[-1], clean.shape[-1])
        enh   = enh[:, :L]
        clean_ = clean[:, :L]

        loss = si_sdr_loss(enh, clean_)
        opt.zero_grad()
        loss.backward()
        opt.step()

        running += loss.item()
    print(f"Epoch {epoch+1}, loss: {running/len(train_dl):.4f}")
