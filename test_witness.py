#!/usr/bin/env python3
import array
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import stat
import tempfile
import unittest
from unittest import mock


SCRIPT = pathlib.Path(__file__).with_name("witness")
LOADER = importlib.machinery.SourceFileLoader("witness_cli", str(SCRIPT))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
WITNESS = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(WITNESS)


class WitnessUnitTests(unittest.TestCase):
    def test_transcription_backend_metadata_is_an_allowlisted_enum(self):
        for backend in ("vulkan", "gguf-cpu", "mixed"):
            safe = WITNESS.sanitize_ledger_record({"transcription_backend": backend})
            self.assertEqual(safe["transcription_backend"], backend)
        safe = WITNESS.sanitize_ledger_record({"transcription_backend": "/private/transcript words"})
        self.assertIsNone(safe["transcription_backend"])

    def _ultra_fixture(self, tmp):
        cli = pathlib.Path(tmp) / "transcribe-cli"
        model = pathlib.Path(tmp) / "ultra.gguf"
        cli.write_text("fixture")
        model.write_text("fixture")
        return mock.patch.multiple(WITNESS, ULTRA_CLI=str(cli), ULTRA_MODEL=str(model), ULTRA_BACKEND="vulkan")

    @staticmethod
    def _ultra_stdout(*items):
        return "\n".join([json.dumps({"type": "batch_header", "load_ms": 900.0})]
                         + [json.dumps(item) for item in items]) + "\n"

    def test_ultra_batch_maps_results_by_path_and_pins_decoder_threads(self):
        with tempfile.TemporaryDirectory() as tmp, self._ultra_fixture(tmp), \
             mock.patch.dict(os.environ, {"LD_LIBRARY_PATH": "/opt/other-ggml"}), \
             mock.patch.object(WITNESS.subprocess, "run") as run:
            jobs = [{"id": 0, "path": f"{tmp}/window-00000.wav"}, {"id": 1, "path": f"{tmp}/window-00001.wav"}]
            run.return_value = mock.Mock(returncode=0, stderr="parakeet: using vulkan backend: Vulkan0", stdout=self._ultra_stdout(
                {"file": jobs[1]["path"], "text": "Bonjour."}, {"file": jobs[0]["path"], "text": "Hello."}))
            results, _, notes = WITNESS.run_ultra(jobs, 4, tmp)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--backend") + 1], "vulkan")
            self.assertEqual(command[command.index("--threads") + 1], "4")
            self.assertIn("--batch-jsonl", command)
            self.assertNotIn("-l", command)
            self.assertNotIn("LD_LIBRARY_PATH", run.call_args.kwargs["env"])
            listed = pathlib.Path(command[command.index("--batch") + 1]).read_text().split()
            self.assertEqual(listed, [jobs[0]["path"], jobs[1]["path"]])
        self.assertEqual(results, {0: {"text": "Hello.", "backend": "vulkan"}, 1: {"text": "Bonjour.", "backend": "vulkan"}})
        self.assertEqual(notes, "")

    def test_ultra_retries_only_failed_windows_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmp, self._ultra_fixture(tmp), \
             mock.patch.object(WITNESS.subprocess, "run") as run:
            jobs = [{"id": 0, "path": f"{tmp}/window-00000.wav"}, {"id": 1, "path": f"{tmp}/window-00001.wav"}]
            run.side_effect = [
                mock.Mock(returncode=0, stderr="", stdout=self._ultra_stdout(
                    {"file": jobs[0]["path"], "text": "Hello."},
                    {"file": jobs[1]["path"], "text": "", "error": "backend error"})),
                mock.Mock(returncode=0, stderr="", stdout=self._ultra_stdout({"file": jobs[1]["path"], "text": "Again."})),
            ]
            results, _, notes = WITNESS.run_ultra(jobs, 4, tmp)
            retry = run.call_args_list[1].args[0]
            self.assertEqual(retry[retry.index("--backend") + 1], "cpu")
            self.assertEqual(pathlib.Path(retry[retry.index("--batch") + 1]).read_text().split(), [jobs[1]["path"]])
        self.assertEqual(results[0]["backend"], "vulkan")
        self.assertEqual(results[1], {"text": "Again.", "backend": "gguf-cpu"})
        self.assertIn("CPU recovery", notes)

    def test_ultra_keeps_partial_text_and_treats_too_short_as_empty(self):
        with tempfile.TemporaryDirectory() as tmp, self._ultra_fixture(tmp), \
             mock.patch.object(WITNESS.subprocess, "run") as run:
            jobs = [{"id": 0, "path": f"{tmp}/a.wav"}, {"id": 1, "path": f"{tmp}/b.wav"}]
            run.return_value = mock.Mock(returncode=0, stderr="", stdout=self._ultra_stdout(
                {"file": jobs[0]["path"], "text": "la la la", "error": "output repetition: decode stopped when the output began repeating itself"},
                {"file": jobs[1]["path"], "text": "", "error": "input audio too short"}))
            results, _, notes = WITNESS.run_ultra(jobs, 4, tmp)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(results[0]["text"], "la la la")
        self.assertEqual(results[1]["text"], "")
        self.assertIn("1 window(s) stopped early", notes)

    def test_ultra_failure_after_cpu_recovery_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp, self._ultra_fixture(tmp), \
             mock.patch.object(WITNESS.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=1, stderr="model load: backend error", stdout="")
            with self.assertRaisesRegex(RuntimeError, "Parakeet Ultra transcription failed"):
                WITNESS.run_ultra([{"id": 0, "path": f"{tmp}/a.wav"}], 4, tmp)
        self.assertEqual(run.call_count, 2)

    def test_missing_ultra_runtime_has_its_own_failure_class(self):
        error = RuntimeError("Parakeet Ultra runtime not found at /fixture/transcribe-cli")
        self.assertEqual(WITNESS.classify_failure(error, "transcription"), "ultra_unavailable")

    def test_every_run_records_the_parakeet_ultra_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = pathlib.Path(tmp) / "runs.jsonl"
            with mock.patch.object(WITNESS, "RUN_LOG_PATH", str(log_path)), \
                 mock.patch.object(WITNESS.sys, "argv", ["witness", f"{tmp}/missing.wav"]), \
                 self.assertRaises(SystemExit):
                WITNESS.main()
            records = [json.loads(line) for line in log_path.read_text().splitlines()]
            self.assertEqual([record["status"] for record in records], ["started", "failed"])
            self.assertEqual({record["transcription_engine"] for record in records}, {"parakeet-ultra"})

    def test_removed_engine_flag_is_rejected(self):
        with mock.patch.object(WITNESS.sys, "stderr"), self.assertRaises(SystemExit):
            WITNESS.build_parser().parse_args(["/a.wav", "--engine", "cohere"])

    def test_configured_diarizer_is_the_default_and_flags_still_win(self):
        # DIARIZER is read from the machine's runtime config at import, so pin it.
        with mock.patch.object(WITNESS, "DIARIZER", "auto"):
            self.assertEqual(WITNESS.build_parser().parse_args(["/a.wav"]).diarizer, "auto")
        with mock.patch.object(WITNESS, "DIARIZER", "cpu"):
            self.assertEqual(WITNESS.build_parser().parse_args(["/a.wav"]).diarizer, "cpu")
            args = WITNESS.build_parser().parse_args(["/a.wav", "--diarizer", "gpu"])
            self.assertEqual(args.diarizer, "gpu")

    def test_markdown_timing_note_names_the_engine(self):
        metadata = {"source": "a.wav", "duration_secs": 2.0, "language": "fr", "language_selection": "explicit",
                    "transcription_engine": "Parakeet Ultra GGUF", "transcription_backend": "vulkan",
                    "transcription_secs": 0.5, "pipeline_secs": 1.0}
        text = WITNESS.markdown_output(metadata, [{"start": 0.0, "end": 1.0, "text": "Bonjour."}], False)
        self.assertIn("Parakeet Ultra chunk windows", text)

    def test_tracked_cli_is_executable(self):
        self.assertTrue(os.access(SCRIPT, os.X_OK))

    def test_overlap_deduplication_ignores_case_and_punctuation(self):
        current, removed = WITNESS.remove_overlap(
            "This is the shared boundary.",
            "the SHARED boundary, then new words arrive.",
        )
        self.assertEqual(removed, 3)
        self.assertEqual(current, "then new words arrive.")

    def test_overlap_deduplication_keeps_unmatched_text(self):
        current, removed = WITNESS.remove_overlap("one speaker ends", "another begins")
        self.assertEqual(removed, 0)
        self.assertEqual(current, "another begins")

    def test_overlap_deduplication_handles_french_morphology_drift(self):
        current, removed = WITNESS.remove_overlap(
            "Bien que l'étiquette ne soit pas payable par l'assurance, bien sûr.",
            "Bien que les étiquettes ne soient pas payables par l'assurance, bien sûr, comme discuté.",
        )
        self.assertGreaterEqual(removed, 10)
        self.assertEqual(current, "comme discuté.")

    def test_turn_normalization_merges_adjacent_same_speaker(self):
        turns = WITNESS.normalize_turns(
            [
                {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
                {"start": 2.3, "end": 4.0, "speaker": "SPEAKER_00"},
                {"start": 4.1, "end": 4.15, "speaker": "SPEAKER_01"},
                {"start": 5.0, "end": 7.0, "speaker": "SPEAKER_01"},
            ],
            10.0,
        )
        self.assertEqual(
            turns,
            [
                {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},
                {"start": 5.0, "end": 7.0, "speaker": "SPEAKER_01"},
            ],
        )

    def test_diarizer_gaps_are_transcribed_as_unknown(self):
        turns = WITNESS.cover_timeline(
            [
                {"start": 1.0, "end": 3.0, "speaker": "SPEAKER_00"},
                {"start": 4.0, "end": 5.0, "speaker": "SPEAKER_01"},
            ],
            [{"start": 0.0, "end": 6.0, "speaker": None}],
        )
        self.assertEqual(
            turns,
            [
                {"start": 0.0, "end": 1.0, "speaker": "UNKNOWN"},
                {"start": 1.0, "end": 3.0, "speaker": "SPEAKER_00"},
                {"start": 3.0, "end": 4.0, "speaker": "UNKNOWN"},
                {"start": 4.0, "end": 5.0, "speaker": "SPEAKER_01"},
                {"start": 5.0, "end": 6.0, "speaker": "UNKNOWN"},
            ],
        )

    def test_silero_silence_gap_is_not_sent_to_asr(self):
        turns = WITNESS.cover_timeline(
            [
                {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
                {"start": 4.0, "end": 6.0, "speaker": "SPEAKER_01"},
            ],
            [
                {"start": 0.0, "end": 2.0, "speaker": None},
                {"start": 4.0, "end": 6.0, "speaker": None},
            ],
        )
        self.assertEqual(len(turns), 2)
        self.assertNotIn("UNKNOWN", {turn["speaker"] for turn in turns})

    def test_diarized_unknown_spans_merge_across_short_vad_pause(self):
        turns = WITNESS.cover_timeline(
            [],
            [
                {"start": 0.0, "end": 1.0, "speaker": None},
                {"start": 1.1, "end": 2.0, "speaker": None},
            ],
        )
        self.assertEqual(
            turns,
            [{"start": 0.0, "end": 2.0, "speaker": "UNKNOWN"}],
        )

    def test_short_vad_pauses_merge_for_asr_context(self):
        regions = WITNESS.merge_speech_regions(
            [
                {"start": 1.0, "end": 10.0, "speaker": None},
                {"start": 10.6, "end": 20.0, "speaker": None},
                {"start": 23.0, "end": 30.0, "speaker": None},
            ]
        )
        self.assertEqual(
            regions,
            [
                {"start": 1.0, "end": 20.0, "speaker": None},
                {"start": 23.0, "end": 30.0, "speaker": None},
            ],
        )

    def test_vad_centiseconds_are_converted_to_seconds(self):
        spans = WITNESS.parse_vad_output(
            "Speech segment 0: start = 39.00, end = 1881.00\n"
            "Speech segment 1: start = 2096.00, end = 2531.00\n"
        )
        self.assertEqual(
            spans,
            [
                {"start": 0.39, "end": 18.81, "speaker": None},
                {"start": 20.96, "end": 25.31, "speaker": None},
            ],
        )

    def test_language_vote_samples_across_file(self):
        language, confidence, probes, votes = WITNESS.choose_detected_language(
            [("fr", 0.95), ("en", 0.90), ("en", 0.85)]
        )
        self.assertEqual(language, "en")
        self.assertAlmostEqual(confidence, 0.875)
        self.assertEqual(probes, 3)
        self.assertEqual(votes, 2)

    def test_language_vote_metadata_distinguishes_unanimous_result(self):
        language, confidence, probes, votes = WITNESS.choose_detected_language(
            [("fr", 0.60), ("fr", 0.55), ("fr", 0.58)]
        )
        self.assertEqual(language, "fr")
        self.assertAlmostEqual(confidence, 0.5766666667)
        self.assertEqual(probes, 3)
        self.assertEqual(votes, 3)

    def test_dominant_unsupported_language_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "dominant auto-detected language is unsupported"):
            WITNESS.choose_detected_language([("es", 0.99), ("en", 0.10), ("es", 0.99)])

    def test_long_span_is_split_with_model_limit_and_overlap(self):
        samples = array.array("h", [100] * (80 * WITNESS.SAMPLE_RATE))
        spans = WITNESS.split_span(samples, 0.0, 80.0, 35.0, 5.0)
        self.assertGreater(len(spans), 1)
        self.assertEqual(spans[0][0], 0.0)
        self.assertEqual(spans[-1][1], 80.0)
        self.assertTrue(all(end - start <= 35.001 for start, end in spans))
        for previous, current in zip(spans, spans[1:]):
            self.assertGreaterEqual(previous[1] - current[0], 0.0)
            self.assertLessEqual(previous[1] - current[0], 5.001)

    def test_auto_diarizer_retries_cpu(self):
        with mock.patch.object(
            WITNESS,
            "_run_diarizer_once",
            side_effect=[
                RuntimeError("GPU init failed"),
                ([{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}], "cpu", 0.5),
            ],
        ) as runner:
            turns, backend, elapsed, first_error = WITNESS.run_diarizer("x.wav", "auto")
        self.assertEqual(runner.call_args_list, [mock.call("x.wav", "gpu"), mock.call("x.wav", "cpu")])
        self.assertEqual(backend, "cpu")
        self.assertEqual(len(turns), 1)
        self.assertGreaterEqual(elapsed, 0.5)
        self.assertLess(elapsed, 0.6)
        self.assertEqual(first_error, "GPU init failed")
        self.assertEqual(WITNESS.classify_fallback(first_error), "initialization_failed")

    def test_srt_contains_speaker_and_monotonic_times(self):
        text = WITNESS.srt_output(
            [
                {"start": 0.0, "end": 1.25, "speaker": "SPEAKER_00", "text": "Hello."},
                {"start": 1.25, "end": 2.0, "speaker": "SPEAKER_01", "text": "Hi."},
            ],
            True,
        )
        self.assertIn("00:00:00,000 --> 00:00:01,250", text)
        self.assertIn("[SPEAKER_00] Hello.", text)
        self.assertIn("00:00:01,250 --> 00:00:02,000", text)

    def test_non_diarized_overlap_cues_are_clamped_after_previous_end(self):
        jobs = [
            {"id": 0, "turn_id": 0, "chunk_id": 0, "start": 0.0, "end": 35.0, "speaker": None},
            {"id": 1, "turn_id": 0, "chunk_id": 1, "start": 30.0, "end": 50.0, "speaker": None},
        ]
        results = {
            0: {"text": "shared boundary"},
            1: {"text": "shared boundary then new text"},
        }
        segments, removed = WITNESS.assemble_segments(
            jobs,
            results,
            [{"start": 0.0, "end": 50.0, "speaker": None}],
            False,
        )
        self.assertEqual(removed, 2)
        self.assertEqual(segments[1]["start"], 35.0)
        self.assertEqual(segments[1]["text"], "then new text")

    def test_explicit_format_controls_default_extension(self):
        self.assertEqual(
            WITNESS.default_output_path("/tmp/call.m4a", True, "srt"),
            "/tmp/call-diarized.srt",
        )
        self.assertEqual(
            WITNESS.default_output_path("/tmp/call.m4a", False, "json"),
            "/tmp/call-transcript.json",
        )

    def test_json_segments_do_not_expose_deleted_temp_paths(self):
        jobs = [
            {
                "id": 0,
                "turn_id": 0,
                "chunk_id": 0,
                "start": 0.0,
                "end": 1.0,
                "speaker": None,
                "path": "/tmp/deleted.wav",
            }
        ]
        segments, _ = WITNESS.assemble_segments(
            jobs,
            {0: {"text": "hello"}},
            [{"start": 0.0, "end": 1.0, "speaker": None}],
            False,
        )
        self.assertNotIn("path", segments[0])

    def test_punctuation_only_hallucination_is_dropped(self):
        segments, _ = WITNESS.assemble_segments(
            [{"id": 0, "turn_id": 0, "chunk_id": 0, "start": 0.0, "end": 1.0, "speaker": None}],
            {0: {"text": "؟"}},
            [{"start": 0.0, "end": 1.0, "speaker": None}],
            False,
        )
        self.assertEqual(segments, [])

    def test_run_ledger_writes_start_and_completed_records_with_mode_0600(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = pathlib.Path(temp_dir) / "state" / "runs.jsonl"
            log_path.parent.mkdir()
            log_path.write_text("", encoding="utf-8")
            log_path.chmod(0o644)
            with mock.patch.object(WITNESS, "RUN_LOG_PATH", str(log_path)):
                ledger = WITNESS.RunLedger("auto", True, "auto", "json")
                self.assertTrue(ledger.start())
                ledger.update(
                    resolved_language="fr",
                    language_confidence=0.975,
                    language_probes=3,
                    language_votes=2,
                    resolved_diarizer="gpu",
                    duration_secs=53.8,
                    speakers_detected=2,
                    unknown_speech_secs=1.25,
                    windows=2,
                    segments=4,
                    transcription_secs=20.1,
                )
                self.assertTrue(ledger.complete())

            records = [json.loads(line) for line in log_path.read_text().splitlines()]
            self.assertEqual([record["status"] for record in records], ["started", "completed"])
            self.assertEqual(records[0]["run_id"], records[1]["run_id"])
            self.assertEqual(records[1]["resolved_language"], "fr")
            self.assertEqual(records[1]["language_probes"], 3)
            self.assertEqual(records[1]["language_votes"], 2)
            self.assertEqual(records[1]["resolved_diarizer"], "gpu")
            self.assertEqual(records[1]["windows"], 2)
            self.assertEqual(set(records[1]), set(WITNESS.LEDGER_FIELDS))
            self.assertEqual(stat.S_IMODE(log_path.stat().st_mode), 0o600)

    def test_run_ledger_structurally_drops_sensitive_and_free_form_values(self):
        secret_path = "/home/person/legal/claim-recording.wav"
        transcript = "private transcript words"
        raw_error = f"decoder crashed while reading {secret_path}: {transcript}"
        record = {
            "run_id": secret_path,
            "timestamp": transcript,
            "status": "failed",
            "stage": raw_error,
            "requested_language": "auto",
            "resolved_language": transcript,
            "diarization_enabled": True,
            "requested_diarizer": "auto",
            "resolved_diarizer": raw_error,
            "fallback_class": raw_error,
            "output_format": "json",
            "failure_class": raw_error,
            "input_path": secret_path,
            "output_path": secret_path + ".json",
            "filename": "claim-recording.wav",
            "text": transcript,
            "helper_log": raw_error,
            "raw_error": raw_error,
        }
        safe = WITNESS.sanitize_ledger_record(record)
        serialized = json.dumps(safe)
        for forbidden in (secret_path, "claim-recording.wav", transcript, raw_error):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(set(safe), set(WITNESS.LEDGER_FIELDS))
        self.assertEqual(safe["stage"], "internal")
        self.assertEqual(safe["failure_class"], "internal_error")
        self.assertEqual(safe["fallback_class"], "other")

    def test_run_ledger_write_failure_is_non_fatal(self):
        ledger = WITNESS.RunLedger("en", False, "disabled", "markdown")
        with mock.patch.object(WITNESS.os, "open", side_effect=PermissionError):
            self.assertFalse(ledger.start())
            self.assertFalse(ledger.complete())

    def test_run_ledger_rejects_fifo_without_blocking_or_mutating_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = pathlib.Path(temp_dir) / "runs.jsonl"
            os.mkfifo(log_path, 0o644)
            original_mode = stat.S_IMODE(log_path.stat().st_mode)
            with mock.patch.object(WITNESS, "RUN_LOG_PATH", str(log_path)):
                self.assertFalse(WITNESS.append_ledger_record({}))
            self.assertTrue(stat.S_ISFIFO(log_path.stat().st_mode))
            self.assertEqual(stat.S_IMODE(log_path.stat().st_mode), original_mode)

    def test_run_ledger_rejects_hardlink_without_mutating_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = pathlib.Path(temp_dir) / "target"
            log_path = pathlib.Path(temp_dir) / "runs.jsonl"
            target.write_text("untouched", encoding="utf-8")
            target.chmod(0o644)
            os.link(target, log_path)
            with mock.patch.object(WITNESS, "RUN_LOG_PATH", str(log_path)):
                self.assertFalse(WITNESS.append_ledger_record({}))
            self.assertEqual(target.read_text(encoding="utf-8"), "untouched")
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o644)
            self.assertEqual(target.stat().st_nlink, 2)

    def test_run_ledger_rejects_final_symlink_without_mutating_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = pathlib.Path(temp_dir) / "target"
            log_path = pathlib.Path(temp_dir) / "runs.jsonl"
            target.write_text("untouched", encoding="utf-8")
            target.chmod(0o644)
            log_path.symlink_to(target)
            with mock.patch.object(WITNESS, "RUN_LOG_PATH", str(log_path)):
                self.assertFalse(WITNESS.append_ledger_record({}))
            self.assertEqual(target.read_text(encoding="utf-8"), "untouched")
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o644)

    def test_run_ledger_rejects_symlinked_parent_component(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            target_parent = root / "real-parent"
            linked_parent = root / "linked-parent"
            target_parent.mkdir()
            linked_parent.symlink_to(target_parent, target_is_directory=True)
            log_path = linked_parent / "runs.jsonl"
            with mock.patch.object(WITNESS, "RUN_LOG_PATH", str(log_path)):
                self.assertFalse(WITNESS.append_ledger_record({}))
            self.assertFalse((target_parent / "runs.jsonl").exists())

    def test_terminal_append_keyboard_interrupt_remains_retryable(self):
        ledger = WITNESS.RunLedger("en", False, "disabled", "markdown")
        with mock.patch.object(
            ledger,
            "_write",
            side_effect=[KeyboardInterrupt(), True],
        ):
            with self.assertRaises(KeyboardInterrupt):
                ledger.complete()
            self.assertFalse(ledger.terminal_written)
            self.assertTrue(ledger.fail(KeyboardInterrupt()))
        self.assertTrue(ledger.terminal_written)
        self.assertEqual(ledger.state["status"], "failed")
        self.assertEqual(ledger.state["failure_class"], "interrupted")

    def test_main_pairs_start_and_failure_for_base_exception_classes(self):
        errors = (
            RuntimeError("raw runtime failure /private/path"),
            SystemExit("raw system exit /private/path"),
            KeyboardInterrupt(),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            for index, error in enumerate(errors):
                with self.subTest(error=type(error).__name__):
                    log_path = pathlib.Path(temp_dir) / f"runs-{index}.jsonl"
                    argv = ["witness", "/private/input-name.wav"]
                    with (
                        mock.patch.object(WITNESS, "RUN_LOG_PATH", str(log_path)),
                        mock.patch.object(WITNESS.sys, "argv", argv),
                        mock.patch.object(WITNESS, "_run_pipeline", side_effect=error),
                        self.assertRaises(type(error)),
                    ):
                        WITNESS.main()
                    records = [json.loads(line) for line in log_path.read_text().splitlines()]
                    self.assertEqual(
                        [record["status"] for record in records],
                        ["started", "failed"],
                    )
                    self.assertEqual(records[0]["run_id"], records[1]["run_id"])
                    self.assertNotIn("/private", log_path.read_text())

    def test_main_logs_normalized_missing_input_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = pathlib.Path(temp_dir) / "private-call-name.wav"
            log_path = pathlib.Path(temp_dir) / "ledger" / "runs.jsonl"
            argv = ["witness", str(missing), "--language", "fr", "--no-diarize"]
            with (
                mock.patch.object(WITNESS, "RUN_LOG_PATH", str(log_path)),
                mock.patch.object(WITNESS.sys, "argv", argv),
                self.assertRaises(SystemExit),
            ):
                WITNESS.main()
            records = [json.loads(line) for line in log_path.read_text().splitlines()]
            self.assertEqual([record["status"] for record in records], ["started", "failed"])
            self.assertEqual(records[-1]["failure_class"], "input_missing")
            self.assertEqual(records[-1]["stage"], "validation")
            self.assertNotIn("private-call-name", log_path.read_text())


if __name__ == "__main__":
    unittest.main()
