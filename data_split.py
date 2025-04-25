import os
import shutil
import random

SRC_BASE = "data"
DEST_BASE = "data/splits"
SPLITS = ["train", "valid", "test"]
SPLIT_RATIOS = [0.8, 0.1, 0.1]
random.seed(42)

mix_folders = {
    "mix1_audio": ("Mix1_clean", ["spk1"]),
    "mix2_audio": ("Mix2_clean", ["spk1", "spk2"]),
    "mix3_audio": ("Mix3_clean", ["spk1", "spk2", "spk3"]),
}

def create_dirs():
    for split in SPLITS:
        for mix in mix_folders:
            os.makedirs(os.path.join(DEST_BASE, split, "Mixture_audio", mix), exist_ok=True)
        for clean in [v[0] for v in mix_folders.values()]:
            os.makedirs(os.path.join(DEST_BASE, split, "Mixture_audio_clean", clean), exist_ok=True)
        for spk in ["spk1", "spk2", "spk3"]:
            os.makedirs(os.path.join(DEST_BASE, split, "Sources", spk), exist_ok=True)
        os.makedirs(os.path.join(DEST_BASE, split, "Sources", "noise_rir"), exist_ok=True)

def split_filenames(files):
    random.shuffle(files)
    total = files.__len__()
    n_train = int(SPLIT_RATIOS[0] * total)
    n_val = int(SPLIT_RATIOS[1] * total)
    return {
        "train": files[:n_train],
        "valid": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:]
    }

def copy_if_exists(src, dst):
    if os.path.exists(src):
        shutil.copy(src, dst)

def split_all():
    for mix_folder, (clean_folder, spk_list) in mix_folders.items():
        mix_dir = os.path.join(SRC_BASE, "Mixture_audio", mix_folder)
        clean_dir = os.path.join(SRC_BASE, "Mixture_audio_clean", clean_folder)
        files = sorted([f for f in os.listdir(mix_dir) if f.endswith(".wav")])

        splits = split_filenames(files)

        for split, split_files in splits.items():
            for fname in split_files:
      
                src_mix = os.path.join(mix_dir, fname)
                dst_mix = os.path.join(DEST_BASE, split, "Mixture_audio", mix_folder, fname)
                copy_if_exists(src_mix, dst_mix)

       
                src_clean = os.path.join(clean_dir, fname)
                dst_clean = os.path.join(DEST_BASE, split, "Mixture_audio_clean", clean_folder, fname)
                copy_if_exists(src_clean, dst_clean)


                for spk in spk_list:
                    spk_fname = fname.replace(".wav", f"_{spk}.wav")
                    src_spk = os.path.join(SRC_BASE, "Sources", spk, spk_fname)
                    dst_spk = os.path.join(DEST_BASE, split, "Sources", spk, spk_fname)
                    copy_if_exists(src_spk, dst_spk)

                rir_fname = fname.replace(".wav", "_noiserir.wav")
                src_rir = os.path.join(SRC_BASE, "Sources", "noise_rir", rir_fname)
                dst_rir = os.path.join(DEST_BASE, split, "Sources", "noise_rir", rir_fname)
                copy_if_exists(src_rir, dst_rir)

if os.path.exists(DEST_BASE):
    shutil.rmtree(DEST_BASE)

create_dirs()
split_all()
print("Split completed")
