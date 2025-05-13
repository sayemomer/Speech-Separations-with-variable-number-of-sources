# Speech Separation with Variable Number of Sources

![Speech Separation Pipeline](./assets/whisper_ecapa_pipeline.png)

## 📋 Overview

This project tackles a critical challenge in conversational AI: handling overlapping speech from an unknown number of speakers. While most speech separation systems only work with a fixed number of speakers, we've developed a novel approach that first counts speakers and then applies the appropriate separation model.

Our system achieves **96.33% accuracy** in speaker counting and produces high-quality separated speech with an average **SDR of 3.96 dB** (excluding single-speaker cases) and **97.62 dB** overall.

## 🔍 Problem Statement

Many conversational AI applications including:
- Speaker diarization
- Automatic speech recognition (ASR)
- Voice command systems
- Meeting transcription

All struggle with overlapping speech. Current separation methods typically assume a fixed number of speakers, limiting their real-world applicability.

## 🚀 Our Approach

We've developed a two-stage solution:

1. **Speaker Count Estimation**: A supervised model predicts the number of speakers (0-3)
2. **Targeted Separation**: Based on the count, we apply the appropriate pre-trained separation model

### Data Pipeline

We created a custom data simulator that:
- Randomly mixes up to three utterances from LibriSpeech-clean-100
- Augments with OpenRIR room impulse responses and noise
- Balances classes across 0/1/2/3 speakers
- Produces ~16 hours of training mixtures with train/validation/test splits

### Speaker Count Models

We benchmarked three embedding approaches:
- **X-Vector**: Traditional speaker embedding
- **ECAPA-TDNN**: Enhanced speaker embedding architecture
- **Whisper-ECAPA-TDNN**: Our novel approach replacing MFCC frontend with Whisper embeddings

### Separation Models

We leveraged SpeechBrain's pre-trained SepFormer models:

- **[speechbrain/sepformer-wsj02mix](https://huggingface.co/speechbrain/sepformer-wsj02mix)**: 
  - Trained on WSJ0-2Mix dataset
  - Optimized for 2-speaker separation
  - Uses transformer-based architecture with attention mechanisms
  - Input: Single-channel mixed audio
  - Output: Two separated audio streams

- **[speechbrain/sepformer-libri3mix](https://huggingface.co/speechbrain/sepformer-libri3mix)**:
  - Trained on LibriMix dataset with 3 speakers
  - Optimized for 3-speaker separation
  - Enhanced transformer architecture for handling more complex mixtures
  - Input: Single-channel mixed audio
  - Output: Three separated audio streams

SepFormer (Separation Transformer) is a state-of-the-art speech separation model that:
- Uses dual-path processing for handling long sequences
- Incorporates self-attention mechanisms for better source modeling
- Achieves separation through mask-based processing in the time-frequency domain
- Surpasses previous methods like Conv-TasNet and DPRNN in separation quality

## 📊 Results

| Model | Clip-level Accuracy |
|-------|---------------------|
| X-Vector | 91.25% |
| ECAPA-TDNN | 94.11% |
| **Whisper-ECAPA-TDNN** | **96.33%** |

Our separation quality metrics:
- Average SDR: 3.96 dB (excluding single-speaker cases)
- Overall SDR: 97.62 dB
- Perceptual improvement confirmed in 86% of clips by 5 listeners

## 🛠️ Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- SpeechBrain
- torchaudio

### Installation

```bash
# Clone the repository
git clone https://github.com/sayemomer/Speech-Separations-with-variable-number-of-sources.git
cd Speech-Separations-with-variable-number-of-sources

# Install dependencies
pip install -r requirements.txt
```

### Dataset Generation

Run the data pipeline to generate the mixtures:

```bash
python data_pipeline.py
python data_split.py
```

### Training and Evaluation

Refer to the notebook `src/main.ipynb` for step-by-step implementation of:
- Data loading and preprocessing
- Model training
- Evaluation metrics
- Results visualization

## 🔮 Future Work

### Model Improvements
- [ ] Extend speaker counting beyond 3 speakers
- [ ] Experiment with different Whisper model sizes (tiny, base, small, medium)
- [ ] Implement real-time processing capabilities
- [ ] Optimize model for edge devices and mobile applications
- [ ] Explore few-shot learning for speaker adaptation

### Dataset Enhancements
- [ ] Include more diverse acoustic environments
- [ ] Add more languages beyond English
- [ ] Create a larger evaluation set for robustness testing
- [ ] Incorporate real-world overlapping speech recordings
- [ ] Generate synthetic data with more varied SNR levels

### Architecture Exploration
- [ ] Test alternative transformer architectures (Conformer, Reformer)
- [ ] Investigate end-to-end joint counting and separation
- [ ] Implement speaker identification alongside separation
- [ ] Explore multi-channel audio support
- [ ] Add diarization capabilities for long-form audio

### Practical Applications
- [ ] Build web demo for interactive testing
- [ ] Create API endpoints for easy integration
- [ ] Develop plug-and-play solution for video conferencing
- [ ] Add support for real-time meeting transcription
- [ ] Implement voice activity detection pre-processing



## 📚 References

1. [SpeechBrain](https://speechbrain.github.io/) - Toolkit for speech processing
2. Whisper: [Radford, A., et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf)
3. ECAPA-TDNN: [Desplanques, B., et al. (2020). ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification](https://arxiv.org/abs/2005.07143)
4. SepFormer: [Subakan, C., et al. (2021). Attention is All You Need in Speech Separation](https://arxiv.org/abs/2010.13154)
5. LibriSpeech: [Panayotov, V., et al. (2015). Librispeech: An ASR corpus based on public domain audio books](https://www.danielpovey.com/files/2015_icassp_librispeech.pdf)
6. OpenRIR: [Ko, T., et al. (2017). A study on data augmentation of reverberant speech for robust speech recognition](https://ieeexplore.ieee.org/document/7953152)
7. LibriMix: [Cosentino, J., et al. (2020). LibriMix: An Open-Source Dataset for Generalizable Speech Separation](https://arxiv.org/abs/2005.11262)

## 👥 Contributors

- [Omer Sayem](www.linkedin.com/in/omer-sayem)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

