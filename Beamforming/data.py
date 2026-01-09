import os, glob
import soundfile as sf

#Dataset manipulation

ROOT = "/kaggle/input/audio-dataset/Audio_Dataset/Audio_Dataset"
SR   = 16000
EXT  = "*.flac"
# =======================
# 1. Pair noisy/clean files
#    (assumes same filename in both folders)
# =======================
def list_pairs(split):
    clean_dir = os.path.join(ROOT, split, "Clean")
    noisy_dir = os.path.join(ROOT, split, "Noisy")
    clean_files = sorted(glob.glob(os.path.join(clean_dir, EXT)))
    pairs = []
    for cpath in clean_files:
        fname = os.path.basename(cpath)
        npath = os.path.join(noisy_dir, fname)
        if os.path.exists(npath):
            pairs.append((npath, cpath))
    return pairs

train_pairs = list_pairs("Train")
val_pairs   = list_pairs("Validation")
test_pairs  = list_pairs("Test")


train_pairs = train_pairs[:100]
val_pairs = val_pairs[:100]
test_pairs = test_pairs[:100]

if __name__ == "__main__":
    # Quick dataset sanity checks when run as a script.
    for split in ["Train", "Validation", "Test"]:
        clean_dir = os.path.join(ROOT, split, "Clean")
        noisy_dir = os.path.join(ROOT, split, "Noisy")
        print(f"=== {split} ===")
        print("Clean dir:", clean_dir, "exists:", os.path.isdir(clean_dir))
        print("Noisy dir:", noisy_dir, "exists:", os.path.isdir(noisy_dir))
        print("Num clean wav:", len(glob.glob(os.path.join(clean_dir, EXT))))
        print("Num noisy wav:", len(glob.glob(os.path.join(noisy_dir, EXT))))
        print()

    # Pick one noisy file and print its shape.
    noisy_dir = os.path.join(ROOT, "Test", "Noisy")
    noisy_files = glob.glob(os.path.join(noisy_dir, "*"))
    if noisy_files:
        audio, sr = sf.read(noisy_files[0], always_2d=True)
        print("Sample rate:", sr)
        print("Audio shape:", audio.shape)
        print("Number of channels:", audio.shape[1])

    print(
        "Num pairs - train/val/test:",
        len(train_pairs),
        len(val_pairs),
        len(test_pairs),
    )
    assert len(train_pairs) > 0, "No training pairs found; check ROOT/path/filenames."
