import numpy as np

try:
    from pystoi.stoi import stoi as _stoi
except Exception:
    _stoi = None

try:
    from pesq import pesq as _pesq
except Exception:
    _pesq = None

# metric calculation

def snr_db(clean, ref):
    L = min(len(clean), len(ref))
    clean = clean[:L]
    ref   = ref[:L]
    noise = ref - clean
    return 10 * np.log10(np.sum(clean**2) / (np.sum(noise**2) + 1e-12))

def mse_db(clean, enh):
    L = min(len(clean), len(enh))
    clean = clean[:L]
    enh   = enh[:L]
    mse = np.mean((clean - enh)**2) + 1e-12
    return 10 * np.log10(mse)


def stoi_score(clean, enh, sr):
    if _stoi is None:
        return None
    L = min(len(clean), len(enh))
    return float(_stoi(clean[:L], enh[:L], sr, extended=False))


def pesq_score(clean, enh, sr, mode="wb"):
    if _pesq is None:
        return None
    L = min(len(clean), len(enh))
    return float(_pesq(sr, clean[:L], enh[:L], mode))
