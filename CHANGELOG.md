# Changelog

## Unreleased

### Added

- NVIDIA support in `install.sh`. With the CUDA 12 or 13 runtime and cuDNN 9,
  it downloads Microsoft's ONNX Runtime 1.24.2 GPU build (pinned by size and
  SHA-256) and runs speaker labels through the CUDA provider, after one real
  check on a sample. With the CUDA toolkit (`nvcc`), it builds transcribe.cpp
  for CUDA, and falls back to Vulkan, then CPU, when that build fails or
  cannot start. `WITNESS_GPU=cuda` stops instead of falling back.
- `WITNESS_DIARIZER_GPU` and the runtime key `diarizer_gpu` choose what the
  speaker-label GPU mode is: `migraphx`, `cuda`, or `cuda-fast`.
- `WITNESS_CUDA_LIB_DIR` and the runtime key `cuda_lib_dir` name CUDA and
  cuDNN directories outside the system loader path.
- `cuda` as a transcription backend (`WITNESS_ULTRA_BACKEND`, runtime key
  `ultra_backend`), with the same CPU retry for failed windows as Vulkan.

Tested on one NVIDIA card: a Tesla T4 with CUDA 13 (Google Colab), where
transcription and speaker labels both ran on the GPU. The CUDA 12 path was
built, and its fallbacks exercised, in NVIDIA's CUDA 12.6 development
container without a GPU.

### Changed

- `speakrs-cli` is built from a newer commit of the fork, which adds the
  `cuda` and `cuda-fast` modes. An existing install rebuilds it on the next
  `install.sh` run.
- `test_witness.py` no longer needs `speakrs-cli` installed: one test failed
  on a machine where Witness had never been installed.
- When speaker labels fall back from the GPU to CPU, the warning and the
  `gpu_diarizer_fallback` metadata now carry the error itself instead of the
  line where `speakrs-cli` names its mode.

## 2026-11-10

First release as its own repository. Witness was developed inside
[onnxruntime-migraphx-rdna4](https://github.com/maherr/onnxruntime-migraphx-rdna4),
and its earlier history is there.

### Added

- `install.sh`: installs Witness on any x86_64 Linux distribution with no
  particular GPU. It prints the missing packages for Fedora, Debian/Ubuntu, or
  Arch, builds transcribe.cpp, whisper.cpp, and `speakrs-cli` at pinned
  commits, downloads the stock ONNX Runtime 1.24.2 and the pinned models with
  size and SHA-256 checks, and writes the runtime config. Transcription uses
  Vulkan when the build files and a hardware device are present, CPU otherwise.
- `WITNESS_DIARIZER` and the runtime key `diarizer` set the default speaker
  label backend. The installer writes `cpu` on a machine without the MIGraphX
  provider, so a run does not start with a failed GPU attempt.
- `WITNESS_ORT_LIB` at install time uses your own ONNX Runtime instead of the
  download. With the MIGraphX provider beside it, speaker labels run on an AMD
  GPU.

### Removed

- The Cohere Transcribe engine. Parakeet Ultra is the only transcription
  model. `--engine`, `WITNESS_ENGINE`, the runtime key `engine`, the
  `WITNESS_COHERE_*` settings, the `witness-cohere-batch` helper, and the older
  installer that set it up are gone.
