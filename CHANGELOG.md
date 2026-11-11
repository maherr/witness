# Changelog

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
