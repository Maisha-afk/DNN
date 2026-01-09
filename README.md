# Beamforming Speech Enhancement

Multi-channel speech enhancement with a CNN mask estimator and MVDR beamforming.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Download the dataset and set the path in `Beamforming/data.py`:

```python
ROOT = "D:/datasets/Audio_Dataset/Audio_Dataset"
```

Expected structure:

```
Audio_Dataset/
  Train/
    Clean/
    Noisy/
  Validation/
    Clean/
    Noisy/
  Test/
    Clean/
    Noisy/
```

## Train

```bash
python Beamforming/train.py
```

## Test

```bash
python Beamforming/test.py
```

If `pystoi` or `pesq` are not installed, those metrics will print as unavailable.
