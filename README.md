# Witness

Witness transcribes English and French audio files on Linux and labels who is speaking. Everything runs on your own machine. It works on CPU on any x86_64 computer, and uses a GPU for transcription when there is one.

```sh
witness meeting.m4a
```

writes `meeting-diarized.md`, a short header followed by the transcript:

```markdown
**SPEAKER_00** [0:00]
Bonjour...

**SPEAKER_01** [0:12]
Oui...
```

Every word in the transcript comes from one model, [Parakeet Ultra](https://huggingface.co/handy-computer/parakeet-ultra-gguf), run through [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp). Witness does not use its timestamps, and it does not label speakers, so Witness supplies timing and speaker labels around it:

1. Convert the input to 16 kHz mono WAV.
2. If language is `auto`, sample high-energy regions across the file, use Whisper only to choose EN or FR for the label in the output, and discard its text.
3. Run Silero VAD to identify speech without generating transcript text.
4. When speaker labels are enabled, run [speakrs](https://github.com/avencera/speakrs) for speaker turns. Silero speech that speakrs missed is retained as `UNKNOWN`.
5. Merge nearby unlabeled speech across short pauses (or all nearby speech when speaker labels are off), then split the timeline into windows of at most 35 seconds with 5 seconds of overlap.
6. Load Parakeet Ultra once, transcribe every window through the same process, and remove overlap repeats. A window that fails on the GPU is retried on CPU with the same model.
7. Emit Markdown, JSON, or SRT using the turn and window times.

## Installing

On any x86_64 Linux distribution:

```sh
git clone https://github.com/maherr/witness
cd witness
bash install.sh
```

The script needs no particular GPU. It first checks for the build tools it needs, and when something is missing it prints the install command for Fedora, Debian/Ubuntu, or Arch and stops. Then it builds and downloads everything into `~/.local/share/witness`:

| Piece | Purpose |
|---|---|
| transcribe.cpp (pinned commit) and Parakeet Ultra Q8_0 (740 MB) | Transcript text |
| whisper.cpp (pinned commit, CPU build), the multilingual base model, Silero VAD v6.2 | The English/French label and speech detection only |
| `speakrs-cli` (pinned commit) and the stock ONNX Runtime 1.24.2 CPU release | Speaker labels |
| `witness` | The command, in `~/.local/bin` |

Every download is pinned to a revision and checked by exact size and SHA-256. Speakrs still fetches its own diarization models from Hugging Face on the first run, and that snapshot is not pinned by this installer. Paths go into `~/.config/witness/runtime.json`; a re-run keeps keys it does not own and skips what is already built. `PREFIX`, `BIN_DIR`, `JOBS`, and `WITNESS_GPU=auto|vulkan|cpu` are the knobs; `SPEAKRS_REPO` and `SPEAKRS_REF` point the build at another speakrs fork or commit.

What runs where:

- **Transcription** uses a GPU through Vulkan when the Vulkan build files (headers, loader, `glslc`) are installed at build time and `transcribe-cli` finds a hardware Vulkan device. Otherwise it runs on CPU, and the installer says so.
- **Speaker labels** run on CPU. The one GPU path for them today is MIGraphX, which is AMD only: see [AMD GPU speaker labels](#amd-gpu-speaker-labels-optional).

### What I tested

All of this is from November 2026, on one machine (Ryzen 7 5800X3D, RX 9070). The test recording is 3 minutes 52 seconds stitched from three public-domain voices (the JFK, `gb0`, and `hp0` samples that whisper.cpp uses), in six alternating stretches.

| Where | Transcription | Speaker labels | Result |
|---|---|---|---|
| Fedora 44, bare container | CPU | CPU | Installs from an empty image using only the commands the script prints. Three speakers found, each voice under its own label |
| Ubuntu 24.04, bare container | CPU | CPU | Same |
| Arch, bare container | CPU | CPU | Same |
| Fedora 44 on the hardware, fresh install prefix | Vulkan, RX 9070 | CPU | Same, and the transcript text matches the CPU run word for word |
| Same machine, `WITNESS_ORT_LIB` pointing at my MIGraphX build | Vulkan, RX 9070 | MIGraphX | Same |

On that recording, on my machine while it was busy with other work, transcription took about 3 seconds on the GPU and 16 to 19 seconds on CPU. Speaker labels took 85 to 105 seconds on CPU and 13 seconds on MIGraphX. So on a CPU-only machine the speaker labels are the slow part, a bit over twice as fast as real time here. `--no-diarize` skips them.

A full install took 10 to 14 minutes and 2 to 2.5 GB.

Not tested: NVIDIA and Intel GPUs (transcription should work through Vulkan, but I have neither), a container with a GPU passed through, any distribution not in the table, Omarchy itself (only its Arch base), and anything that is not x86_64. There is no CUDA path yet, for transcription or for speaker labels. If you run it somewhere new, an issue with the result is welcome either way.

### AMD GPU speaker labels (optional)

To run the speaker labels on an AMD GPU, give the installer an ONNX Runtime 1.24.2 that has the MIGraphX provider:

```sh
WITNESS_ORT_LIB=/path/to/libonnxruntime.so.1.24.2 bash install.sh
```

When `libonnxruntime_providers_migraphx.so` sits beside that library, the installer leaves the diarizer on `auto` (GPU first, CPU fallback) instead of `cpu`. Building that ONNX Runtime is a separate project: [onnxruntime-migraphx-rdna4](https://github.com/maherr/onnxruntime-migraphx-rdna4) has the build script and patches I used for RDNA 4 on Fedora 43 with ROCm 6.4.2, and a status note for Fedora 44. If MIGraphX lives outside the system library path, set the runtime keys `migraphx_lib_dir` and `migraphx_cache_dir` yourself; the installer keeps them.

## Usage

```sh
witness input.m4a
witness input.m4a --language fr
witness input.m4a --no-diarize
witness input.m4a --output transcript.json
witness input.m4a --output transcript.srt
witness input.m4a --diarizer cpu
```

Defaults:

- language: use one high-energy probe for short files or three distributed probes for long files and aggregate the EN/FR result. Parakeet Ultra detects the spoken language itself, so this only labels the output
- speaker labels: enabled. `install.sh` sets the diarizer to `cpu` on a machine without the MIGraphX provider; `auto` tries the MIGraphX GPU first with an automatic CPU retry. Choose with `--diarizer`, `WITNESS_DIARIZER`, or the runtime key `diarizer`
- output: `<stem>-diarized.md`
- windows: 35 seconds with 5 seconds overlap
- speech regions: Silero VAD v6.2; no VAD or Whisper words enter the transcript

Use `--language en` or `--language fr` when you already know the language. `auto` approximates the file's dominant supported language from distributed probes; it does not route each sentence independently.

`--turbo`, `--model`, and `--serial` remain accepted for old command lines but are ignored. Transcript text always comes from Parakeet Ultra, never from Whisper.

## Parakeet Ultra

`install.sh` builds the runtime, downloads the model, and records both paths in the runtime config. It builds transcribe.cpp from commit `5d66f60257a75e9bdbddc72dc0887b32e9d4841a` (on `main`, after the 0.3.1 release), with the Vulkan backend when the build files are there. `WITNESS_ULTRA_BACKEND=cpu` (runtime key `ultra_backend`) runs it on CPU deliberately. The decoder runs on CPU threads even on Vulkan; Witness passes four (`--threads`), because the runtime default of eight stalled under desktop load on my machine.

If the runtime or the model is missing, Witness stops with an error. It never falls back to another model, so one run carries one model's text.

Measured through Witness on my machine (RX 9070, Vulkan) in October 2026, word error rate, lower is better:

| Set | Plain | With speaker labels |
|---|---:|---:|
| Earnings-21, 4 calls, 128 min (English) | 9.03% | 9.42% |
| CallFriend Québécois, 4 channels, 119 min (French phone) | 53.88% | 56.27% |

Four recordings per set, so read these as a direction, not a benchmark. The CallFriend references are imperfect and cover one side of each call, which is why the French numbers are so high. I measured these with my own shared-library build of the same transcribe.cpp commit, not with the static build `install.sh` makes, and I have not re-measured them since. The scoring scripts are not in this repository yet.

## Output

Markdown includes the provenance of each stage:

```markdown
# Diarized Transcript, example.m4a

**Duration:** 3:00
**Language:** fr (auto-detected p=0.991; Whisper words discarded)
**Transcription:** Parakeet Ultra GGUF, vulkan
**Speakers detected:** 2
**Diarization:** speakrs, cpu
**Timestamp precision:** speaker-turn boundaries from speakrs; Parakeet Ultra word times are not used

---

**SPEAKER_00** [0:00]
Bonjour...

**SPEAKER_01** [0:12]
Oui...
```

The example is abridged: the real header also carries transcription and diarization times. The transcription line names the backend that ran (`vulkan`, `gguf-cpu`, or `mixed` when some windows were recovered on CPU).

JSON contains the same metadata plus structured segments. SRT uses speaker-turn boundaries when speaker labels are on and transcription windows otherwise. Those times are navigation cues, not word alignment.

## Operational ledger

Once command-line arguments parse successfully, Witness appends a start record
and a terminal `completed` or `failed` record to
`~/.local/state/witness/runs.jsonl`. Argument-parser exits such as invalid flags,
a missing required argument, or `--help` can occur before the ledger starts.
The file is forced to mode `0600`, and ordinary logging I/O failures never
interrupt transcription.

The ledger is deliberately metadata-only. It records a random run ID, UTC
timestamps, normalized pipeline stage/status, EN/FR selection and confidence,
diarization backend/fallback class, transcription backend, duration,
speaker and `UNKNOWN` coverage, window/segment counts, output format, and stage
timings. It is a local file: Witness sends it nowhere. It never records input
or output paths, filenames, transcript text, helper logs, or raw error strings.
All string fields except the random ID and timestamp are fixed enums, and the
serializer drops unknown fields.

## Runtime overrides

| Variable | Purpose |
|---|---|
| `WITNESS_RUNTIME_CONFIG` | Installer-written runtime path map (default `~/.config/witness/runtime.json`) |
| `WITNESS_ULTRA_CLI` | transcribe.cpp `transcribe-cli` path; runtime key `ultra_cli` |
| `WITNESS_ULTRA_MODEL` | Parakeet Ultra GGUF path; runtime key `ultra_model` |
| `WITNESS_ULTRA_BACKEND` | `vulkan` (default) or `cpu`; runtime key `ultra_backend` |
| `WITNESS_WHISPER_CLI` | Whisper language-classifier binary |
| `WITNESS_LANGUAGE_MODEL` | Multilingual Whisper model used only for EN/FR detection |
| `WITNESS_VAD_CLI` | Standalone whisper.cpp Silero VAD helper |
| `WITNESS_VAD_MODEL` | Silero VAD model used only for speech-region detection |
| `WITNESS_SPEAKRS_CLI` | `speakrs-cli` path |
| `WITNESS_DIARIZER` | `auto` (default), `gpu`, or `cpu`; runtime key `diarizer` |
| `WITNESS_ORT_LIB` | `libonnxruntime.so.1.24.2` path used by speakrs |
| `WITNESS_MIGRAPHX_LIB_DIR` | Additional MIGraphX library directory |
| `ORT_MIGRAPHX_MODEL_CACHE_PATH` | MIGraphX compiled-model cache |
| `WITNESS_RUN_LOG` | Privacy-safe operational ledger path (default `~/.local/state/witness/runs.jsonl`) |

With no runtime config and no override, Witness falls back to the paths of my own machine's older layout. `install.sh` always writes the config, so that only matters if you install the pieces by hand.

## Verification

```sh
python3 -m unittest -v test_witness.py
witness path/to/english-two-speaker.wav --language en
witness path/to/french.wav --language fr --no-diarize
```

The 40-test unit suite covers long-window splitting, overlap deduplication (including French morphology drift), speaker-turn normalization, short-pause context merging with and without speaker labels, Silero time parsing and speakrs gap coverage, supported-language voting, GPU-to-CPU fallback, the configured diarizer default, Parakeet Ultra batch mapping and CPU retry, output extensions, JSON hygiene, SRT timing, and ledger privacy/permissions.

## Known limitations

- Witness uses no model timestamps. Its timestamps come from `speakrs` turns or 35 second windows.
- `UNKNOWN` labels mean Silero detected speech that speakrs did not assign to a speaker; the audio was still transcribed.
- Parakeet Ultra takes no language hint. On short French turns it sometimes writes English: on the four Quebec French phone calls above, run with speaker labels, 12 to 21% of its segments read as English, mostly turns under a second.
- Overlapping speakers remain difficult. Each overlapping turn contains the same mixed waveform, so both transcription jobs can hear both voices.
- Speaker labels are local to each file (`SPEAKER_00`, `SPEAKER_01`, and so on). There is no cross-file voice identification.
- The language label supports English and French only, and a file whose dominant language is neither stops with an error. Pass an explicit language when a short or noisy file is misclassified.
- SRT timing is coarse. Use a separate forced aligner if word-level timing is required.
- The current wrapper converts and stages the whole recording before inference. Very long files require about 115 MB RAM/hour for PCM plus roughly 250 MB/hour of temporary WAV storage with default overlap, and a persistent failed window still fails the file rather than emitting a partial transcript.

## History

Witness started inside [onnxruntime-migraphx-rdna4](https://github.com/maherr/onnxruntime-migraphx-rdna4), where it was the application for the AMD GPU speaker-label work, and its earlier history is there. Until November 2026 it could also transcribe with Cohere Transcribe; that engine is gone, and Parakeet Ultra is the only one.

## License

MIT. See `THIRD_PARTY_LICENSES.md` for model and dependency attribution.
