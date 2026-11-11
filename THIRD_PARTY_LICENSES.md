# Third-Party Licenses

Witness builds on the following open-source components. This is the key
license and attribution inventory, not a replacement for the license files in
each dependency. This repository does not bundle third-party source, model
weights, or prebuilt binaries: `install.sh` clones each dependency at a pinned
commit, builds it on your machine, and downloads the models.

## Code Dependencies

### transcribe.cpp
- **License**: MIT
- **Copyright**: 2026 The transcribe.cpp authors
- **Source**: https://github.com/handy-computer/transcribe.cpp
- **Usage**: Parakeet Ultra GGUF inference through `transcribe-cli`, the source of every transcript word

### whisper.cpp
- **License**: MIT
- **Copyright**: 2023-2026 The ggml authors
- **Source**: https://github.com/ggerganov/whisper.cpp
- **Usage**: English/French language classification and the Silero VAD helper only; its transcript words are discarded

### speakrs
- **License**: Apache-2.0
- **Copyright**: Praveen Perera
- **Source**: https://github.com/avencera/speakrs
- **Usage**: speaker diarization pipeline (segmentation, embedding, clustering). `install.sh` builds the command-line tool from https://github.com/maherr/speakrs, a fork that adds a MIGraphX execution mode

### ONNX Runtime
- **License**: MIT
- **Copyright**: Microsoft Corporation
- **Source**: https://github.com/microsoft/onnxruntime
- **Usage**: inference runtime for speakrs. `install.sh` downloads Microsoft's unmodified 1.24.2 Linux x64 release archive, pinned by size and SHA-256

### FFmpeg
- **License**: LGPL-2.1-or-later or GPL-2.0-or-later, depending on the distribution's build
- **Source**: https://ffmpeg.org/
- **Usage**: input conversion, called as the `ffmpeg` command from your distribution's package; not built or downloaded here

## Model Weights

Model weights are not bundled in this repository. `install.sh` downloads the
pinned Parakeet Ultra Q8_0, Whisper base, and Silero VAD artifacts from Hugging
Face and verifies their sizes and SHA-256 digests. Speakrs downloads its own
diarization models through its ModelManager on the first run, as converted
copies from its own Hugging Face model repository. The installs I tested needed
no Hugging Face account.

### Parakeet Ultra (Q8_0 GGUF conversion)
- **License**: CC-BY-4.0
- **Source model**: https://huggingface.co/moondream/parakeet-ultra
- **GGUF conversion**: https://huggingface.co/handy-computer/parakeet-ultra-gguf (converted and quantized for transcribe.cpp; not retrained or fine-tuned, per its NOTICE)
- **Usage**: transcript text; downloaded by `install.sh`, not bundled here

### Whisper base multilingual model (GGML conversion)
- **License**: MIT
- **Copyright**: OpenAI; GGML conversion distributed by whisper.cpp
- **Source model**: https://github.com/openai/whisper
- **Converted artifact**: https://huggingface.co/ggerganov/whisper.cpp
- **Usage**: English/French language classification only; no Whisper transcript text is emitted

### Silero VAD v6.2 (GGML conversion)
- **License**: MIT
- **Copyright**: 2020-present Silero Team
- **Source**: https://github.com/snakers4/silero-vad
- **Converted artifact**: https://huggingface.co/ggml-org/whisper-vad
- **Usage**: speech-region detection only; it does not generate transcript text

### pyannote segmentation-3.0
- **License**: MIT
- **Copyright**: Herve Bredin / IRIT / pyannote.audio
- **Source**: https://huggingface.co/pyannote/segmentation-3.0
- **Usage**: speaker segmentation model used by speakrs (speech activity detection and speaker change detection)
- **Access**: the original is gated on Hugging Face; speakrs fetches its own converted copy

### WeSpeaker ResNet34 (VoxCeleb)
- **License**: CC-BY-4.0 (Creative Commons Attribution 4.0 International)
- **Copyright**: WeNet / VoxCeleb authors
- **Source**: https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
- **Training data**: VoxCeleb (CC-BY-4.0, https://mm.kaist.ac.kr/datasets/voxceleb/)
- **Usage**: speaker embedding extraction in speakrs
- **Attribution requirement**: CC-BY-4.0 requires attribution. See below.

## Required Attribution (CC-BY-4.0)

The WeSpeaker ResNet34 embedding model is trained on the VoxCeleb dataset,
licensed under CC-BY-4.0. Per the license terms, we acknowledge:

> VoxCeleb: A Large-scale Speaker Verification Dataset.
> A. Nagrani, J. S. Chung, A. Zisserman.
> INTERSPEECH, 2017.
>
> VoxCeleb2: Deep Speaker Recognition.
> J. S. Chung, A. Nagrani, A. Zisserman.
> INTERSPEECH, 2018.
>
> WeSpeaker: A Research and Production oriented Speaker Embedding Learning Toolkit.
> https://github.com/wenet-e2e/wespeaker

Any redistribution of this project or derivative works that include or invoke
the WeSpeaker model must include this attribution.
