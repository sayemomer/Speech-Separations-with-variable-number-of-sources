#Code for Experiment
import os
import shutil
import random
import numpy as np
import torch
import torchaudio
import soundfile as sf
from scipy.signal import fftconvolve
from torchaudio.datasets import LIBRISPEECH

BASE = "data/RIRS_NOISES"
noise_paths = []
rir_paths   = []

ps_dir = os.path.join(BASE, "pointsource_noises")
for root, _, files in os.walk(ps_dir):
    for f in files:
        if f.lower().endswith(".wav"):
            noise_paths.append(os.path.join(root, f))

rr_dir = os.path.join(BASE, "real_rirs_isotropic_noises")
for root, _, files in os.walk(rr_dir):
    for f in files:
        # skip non‑audio files
        if not f.lower().endswith(".wav"):
            continue
        path = os.path.join(root, f)
       
        if "rir" in f.lower():
            rir_paths.append(path)
        else:
            noise_paths.append(path)

sr_dir = os.path.join(BASE, "simulated_rirs")
for root, _, files in os.walk(sr_dir):
    for f in files:
        if f.lower().endswith(".wav"):
            rir_paths.append(os.path.join(root, f))

print(f"Collected {len(noise_paths)} number of noise files")
print(f"Collected {len(rir_paths)} number of RIR files")

os.makedirs("data/rir",   exist_ok=True)
os.makedirs("data/noise", exist_ok=True)


for src_path in noise_paths:
    fname = os.path.basename(src_path)
    dst   = os.path.join("data/noise", fname)
    shutil.copy(src_path, dst)

for src_path in rir_paths:
    fname = os.path.basename(src_path)
    dst   = os.path.join("data/rir", fname)
    shutil.copy(src_path, dst)

print("stored files into data/noise and data/rir folder")


Libri_dir      = "data/"
Noise_dir      = "data/noise"
RIR_dir        = "data/rir"
Mixture_base   = "data/Mixture_audio"
Clean_base_dir = "data/Mixture_audio_clean"
Source_dir     = "data/Sources"

Total_aud_per_class = 1000  
Max_spk             = 3
Max_dur_sec         = 5.0
SR                  = 16000
Target_Len          = int(Max_dur_sec * SR)

Mixture_dirs = {
    n: os.path.join(Mixture_base, f"mix{n}_audio")
    for n in range(1, Max_spk + 1)
}
for d in Mixture_dirs.values():
    os.makedirs(d, exist_ok=True)


clean_dirs = {
    n: os.path.join(Clean_base_dir, f"Mix{n}_clean")
    for n in range(1, Max_spk + 1)
}
for d in clean_dirs.values():
    os.makedirs(d, exist_ok=True)


os.makedirs(Source_dir, exist_ok=True)
noise_rir_dir = os.path.join(Source_dir, "noise_rir")
os.makedirs(noise_rir_dir, exist_ok=True)
Spk_Dirs = {
    i: os.path.join(Source_dir, f"spk{i+1}")
    for i in range(Max_spk)
}
for d in Spk_Dirs.values():
    os.makedirs(d, exist_ok=True)

dataset   = LIBRISPEECH(Libri_dir, url="train-clean-100", download=False)
num_utts  = dataset.__len__()


noise_paths = [os.path.join(Noise_dir, f)
               for f in os.listdir(Noise_dir) if f.endswith(".wav")]
noises = []
for p in noise_paths:
    try:
        wav, _ = torchaudio.load(p)
        arr = wav.numpy()
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        else:
            arr = arr.squeeze()
        noises.append(arr.astype(np.float32))
    except Exception:
        print(f"Skipping broken noise: {p}")

rir_paths = [os.path.join(RIR_dir, f)
             for f in os.listdir(RIR_dir) if f.endswith(".wav")]
rirs = []
for p in rir_paths:
    try:
        wav, _ = torchaudio.load(p)
        arr = wav.numpy()
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        else:
            arr = arr.squeeze()
        rirs.append(arr.astype(np.float32))
    except Exception:
        print(f"Skipping broken RIR: {p}")

print(f"Loaded {rirs.__len__()} usable RIRs")
print(f"Loaded {noises.__len__()} usable noises")


for n_spk in range(1, Max_spk + 1):
    for idx in range(Total_aud_per_class):
        # pick n_spk distinct sources
        sources = []
        seen = set()
        while sources.__len__() < n_spk:
            wav, _, _, spk_id, _, _ = dataset[random.randrange(num_utts)]
            if spk_id in seen:
                continue
            seen.add(spk_id)
            sig = wav.squeeze().numpy()
            if sig.shape[0] > Target_Len:
                start = random.randint(0, sig.shape[0] - Target_Len)
                sig = sig[start:start + Target_Len]
            else:
                sig = np.pad(sig, (0, Target_Len - sig.shape[0]))
            sources.append(sig.astype(np.float32))

   
        clean_mix = np.sum(sources, axis=0).astype(np.float32)
        clean_path = os.path.join(clean_dirs[n_spk], f"mix{n_spk}_{idx}.wav")
        torchaudio.save(
            clean_path,
            torch.from_numpy(clean_mix).unsqueeze(0),
            sample_rate=SR
        )

 
        mixture    = clean_mix.copy()
        noise_comp = np.zeros(Target_Len, dtype=np.float32)


        if noises and random.random() < 0.5:
            samples = random.choice(noises)
            if samples.shape[0] < Target_Len:
                samples = np.pad(samples, (0, Target_Len - samples.shape[0]))
            else:
                start = random.randint(0, samples.shape[0] - Target_Len)
                samples = samples[start:start + Target_Len]
            e_s = np.mean(mixture**2) + 1e-9
            e_n = np.mean(samples**2) + 1e-9
            snr = random.uniform(0, 10)
            samples *= np.sqrt(e_s / (10**(snr/10) * e_n))
            mixture += samples
            noise_comp = samples.copy()

  
        if rirs and random.random() < 0.5:
            h = random.choice(rirs)
            mixture    = fftconvolve(mixture, h)[:Target_Len]
            noise_comp = fftconvolve(noise_comp, h)[:Target_Len]
            sources    = [fftconvolve(s, h)[:Target_Len] for s in sources]

        peak = np.max(np.abs(mixture)) + 1e-9
        mixture    = mixture    / peak * 0.9
        noise_comp = noise_comp / peak * 0.9
        sources    = [s / peak * 0.9 for s in sources]


        mix_dir = Mixture_dirs[n_spk]
        mix_path = os.path.join(mix_dir, f"mix{n_spk}_{idx}.wav")
        torchaudio.save(
            mix_path,
            torch.from_numpy(mixture).unsqueeze(0),
            sample_rate=SR
        )


        nr_path = os.path.join(noise_rir_dir, f"mix{n_spk}_{idx}_noiserir.wav")
        torchaudio.save(
            nr_path,
            torch.from_numpy(noise_comp).unsqueeze(0),
            sample_rate=SR
        )


        for i, s in enumerate(sources):
            spk_dir = Spk_Dirs[i]
            spk_path = os.path.join(spk_dir, f"mix{n_spk}_{idx}_spk{i+1}.wav")
            torchaudio.save(
                spk_path,
                torch.from_numpy(s).unsqueeze(0),
                sample_rate=SR
            )

print("Stored all mixtures and sources.")



numb_files = (sum( 1 for fn in os.listdir("data/Mixture_audio/mix1_audio"))
+sum( 1 for fn in os.listdir("data/Mixture_audio/mix2_audio"))
+sum( 1 for fn in os.listdir("data/Mixture_audio/mix3_audio")))

print(f"Total mixture audio: {numb_files}")


numb_files1 = (sum( 1 for fn in os.listdir("data/Mixture_audio_clean/Mix1_clean"))
+sum( 1 for fn in os.listdir("data/Mixture_audio_clean/Mix2_clean"))
+sum( 1 for fn in os.listdir("data/Mixture_audio_clean/Mix3_clean")))

print(f"Total mixture audio clean: {numb_files1}")


noise_rir_dir = "data/Sources/noise_rir"
num_files2 = sum(1 for fn in os.listdir(noise_rir_dir))
print(f"Total noise_rir audio files: {num_files2}")


spk1 = "data/Sources/spk1"
spk11 = sum( 1 for fn in os.listdir(spk1))
print(f"Total speaker 1: {spk11}")

spk2 = "data/Sources/spk2"
spk22 = sum( 1 for fn in os.listdir(spk2))
print(f"Total speaker 2: {spk22}")

spk3 = "data/Sources/spk3"
spk33 = sum( 1 for fn in os.listdir(spk3))
print(f"Total speaker 3: {spk33}")
