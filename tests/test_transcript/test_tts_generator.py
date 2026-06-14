"""
Tests for the TTS podcast generator.

Run with: pytest tests/test_tts_generator.py -v
"""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transcript_pipeline.tts_generator import (
    parse_speaker_segments_from_md,
    _resolve_voice,
    DEFAULT_VOICE_MAP,
    DEFAULT_FALLBACK_VOICE,
    DEFAULT_PAUSE_MS,
    HAS_EDGE_TTS,
    HAS_PYDUB,
    HAS_TTS,
)


# ── Sample Markdown for testing ────────────────────────────────────────────────

SAMPLE_MD_WITH_SPEAKERS = """# Test Podcast

> **Source Video ID:** `Test_Podcast`

---

## Transcript (with speaker labels)

**SPEAKER_01:** Hello and welcome to the show.

**SPEAKER_00:** Thanks for having me here today. It's great to be back.

**SPEAKER_01:** Let's dive into our first topic.

**SPEAKER_00:** Sounds good. I've been looking forward to this.

---

*This document was auto-generated from the audio transcript and preserves the original content in full.*
"""

SAMPLE_MD_NAMED_SPEAKERS = """# Test Podcast

---

## Transcript (with speaker labels)

**Alice:** Hello everyone, welcome.

**Bob:** Thanks Alice. Great to be here.

**Alice:** Let's get started.

---

*Auto-generated.*
"""

SAMPLE_MD_NO_SPEAKERS = """# Test Transcript

---

## Transcript

This is a standard transcript without speaker labels.
It has multiple paragraphs.

This is the second paragraph of the transcript.

---

*Auto-generated.*
"""


# ── parse_speaker_segments_from_md ─────────────────────────────────────────────


class TestParseSpeakerSegmentsFromMd:
    """Tests for parsing speaker-labeled Markdown."""

    def test_parse_basic_speakers(self):
        segments = parse_speaker_segments_from_md(SAMPLE_MD_WITH_SPEAKERS)
        assert len(segments) == 4
        assert segments[0]["speaker"] == "SPEAKER_01"
        assert "Hello and welcome" in segments[0]["text"]
        assert segments[1]["speaker"] == "SPEAKER_00"
        assert "Thanks for having me" in segments[1]["text"]

    def test_parse_named_speakers(self):
        segments = parse_speaker_segments_from_md(SAMPLE_MD_NAMED_SPEAKERS)
        assert len(segments) == 3
        assert segments[0]["speaker"] == "Alice"
        assert "Hello everyone" in segments[0]["text"]
        assert segments[1]["speaker"] == "Bob"
        assert "Great to be here" in segments[1]["text"]

    def test_parse_no_speakers_fallback(self):
        segments = parse_speaker_segments_from_md(SAMPLE_MD_NO_SPEAKERS)
        # Should fall back to SPEAKER_DEFAULT with paragraph grouping
        assert len(segments) >= 1
        assert segments[0]["speaker"] == "SPEAKER_DEFAULT"

    def test_empty_markdown(self):
        segments = parse_speaker_segments_from_md("")
        assert segments == []

    def test_frontmatter_only(self):
        md = "# Title\n\n---\n\n---\n\n*Footer*\n"
        segments = parse_speaker_segments_from_md(md)
        # No transcript body — should return empty
        assert len(segments) == 0

    def test_single_speaker(self):
        md = """# Solo Podcast

---

## Transcript (with speaker labels)

**SPEAKER_00:** I'm talking to myself today.

**SPEAKER_00:** Still just me here.

---

*Done.*
"""
        segments = parse_speaker_segments_from_md(md)
        assert len(segments) == 2
        assert all(s["speaker"] == "SPEAKER_00" for s in segments)

    def test_skip_blank_segments(self):
        md = """# Test

---

## Transcript (with speaker labels)

**SPEAKER_00:** First point.

**SPEAKER_01:**

**SPEAKER_00:** Third point.

---

*Done.*
"""
        segments = parse_speaker_segments_from_md(md)
        # The empty SPEAKER_01 line should be skipped
        assert len(segments) == 2
        assert segments[0]["speaker"] == "SPEAKER_00"
        assert segments[1]["speaker"] == "SPEAKER_00"


# ── _resolve_voice ─────────────────────────────────────────────────────────────


class TestResolveVoice:
    """Tests for voice resolution logic."""

    def test_builtin_default_speaker_00(self):
        voice = _resolve_voice("SPEAKER_00")
        assert voice == DEFAULT_VOICE_MAP["SPEAKER_00"]
        assert voice == "en-US-GuyNeural"

    def test_builtin_default_speaker_01(self):
        voice = _resolve_voice("SPEAKER_01")
        assert voice == DEFAULT_VOICE_MAP["SPEAKER_01"]
        assert voice == "en-US-JennyNeural"

    def test_fallback_for_unknown_speaker(self):
        voice = _resolve_voice("SPEAKER_99")
        assert voice == DEFAULT_FALLBACK_VOICE

    def test_explicit_map_overrides_default(self):
        custom_map = {"SPEAKER_00": "en-GB-RyanNeural"}
        voice = _resolve_voice("SPEAKER_00", voice_map=custom_map)
        assert voice == "en-GB-RyanNeural"

    def test_explicit_map_partial_fallback(self):
        """SPEAKER_01 not in custom map — should fallback to default chain."""
        custom_map = {"SPEAKER_00": "en-GB-RyanNeural"}
        voice = _resolve_voice("SPEAKER_01", voice_map=custom_map)
        assert voice == "en-US-JennyNeural"  # from DEFAULT_VOICE_MAP

    def test_named_speaker_in_custom_map(self):
        custom_map = {"Alice": "en-US-JennyNeural", "Bob": "en-US-GuyNeural"}
        assert _resolve_voice("Alice", voice_map=custom_map) == "en-US-JennyNeural"
        assert _resolve_voice("Bob", voice_map=custom_map) == "en-US-GuyNeural"

    def test_env_var_tts_default_voice_00(self, monkeypatch):
        monkeypatch.setenv("TTS_DEFAULT_VOICE_00", "en-GB-RyanNeural")
        voice = _resolve_voice("SPEAKER_00")
        assert voice == "en-GB-RyanNeural"

    def test_env_var_tts_default_voice_01(self, monkeypatch):
        monkeypatch.setenv("TTS_DEFAULT_VOICE_01", "en-AU-NatashaNeural")
        voice = _resolve_voice("SPEAKER_01")
        assert voice == "en-AU-NatashaNeural"

    def test_env_var_tts_voice_map_json(self, monkeypatch):
        monkeypatch.setenv(
            "TTS_VOICE_MAP_JSON",
            '{"SPEAKER_00":"en-US-EricNeural","SPEAKER_01":"en-GB-SoniaNeural"}',
        )
        assert _resolve_voice("SPEAKER_00") == "en-US-EricNeural"
        assert _resolve_voice("SPEAKER_01") == "en-GB-SoniaNeural"

    def test_env_var_voice_map_json_partial(self, monkeypatch):
        """Only SPEAKER_00 in env — SPEAKER_01 should use built-in default."""
        monkeypatch.setenv(
            "TTS_VOICE_MAP_JSON",
            '{"SPEAKER_00":"en-US-EricNeural"}',
        )
        assert _resolve_voice("SPEAKER_00") == "en-US-EricNeural"
        assert _resolve_voice("SPEAKER_01") == "en-US-JennyNeural"

    def test_explicit_map_highest_priority(self, monkeypatch):
        """Explicit voice_map should override env vars and built-ins."""
        monkeypatch.setenv("TTS_DEFAULT_VOICE_00", "en-GB-RyanNeural")
        monkeypatch.setenv("TTS_VOICE_MAP_JSON", '{"SPEAKER_00":"en-US-EricNeural"}')
        custom_map = {"SPEAKER_00": "en-AU-WilliamNeural"}
        voice = _resolve_voice("SPEAKER_00", voice_map=custom_map)
        assert voice == "en-AU-WilliamNeural"


# ── Module-level flags ─────────────────────────────────────────────────────────


class TestAvailabilityFlags:
    """Test that module-level flags are set correctly."""

    def test_has_edge_tts_is_bool(self):
        assert isinstance(HAS_EDGE_TTS, bool)

    def test_has_pydub_is_bool(self):
        assert isinstance(HAS_PYDUB, bool)

    def test_has_tts_is_bool(self):
        assert isinstance(HAS_TTS, bool)

    def test_has_tts_consistent(self):
        assert HAS_TTS == (HAS_EDGE_TTS and HAS_PYDUB)

    def test_defaults_are_strings(self):
        assert isinstance(DEFAULT_FALLBACK_VOICE, str)
        assert len(DEFAULT_FALLBACK_VOICE) > 0

    def test_default_pause_is_int(self):
        assert isinstance(DEFAULT_PAUSE_MS, int)
        assert DEFAULT_PAUSE_MS > 0
        assert DEFAULT_PAUSE_MS <= 5000


# ── Integration tests (require edge-tts + pydub) ───────────────────────────────


@pytest.mark.skipif(not HAS_TTS, reason="edge-tts and pydub required")
class TestGeneratePodcastIntegration:
    """End-to-end tests that actually generate audio."""

    def test_generate_podcast_from_segments(self, tmp_path):
        from transcript_pipeline.tts_generator import generate_podcast_sync

        segments = [
            {"speaker": "SPEAKER_01", "text": "Hello, this is a test."},
            {"speaker": "SPEAKER_00", "text": "Yes, it is a test of the TTS system."},
        ]
        output = tmp_path / "test_podcast.mp3"

        result_path = generate_podcast_sync(
            segments=segments,
            output_path=str(output),
        )

        assert os.path.isfile(result_path)
        assert os.path.getsize(result_path) > 0

    def test_generate_podcast_from_md(self, tmp_path):
        from transcript_pipeline.tts_generator import generate_podcast_sync

        md_file = tmp_path / "test.md"
        md_file.write_text(SAMPLE_MD_WITH_SPEAKERS, encoding="utf-8")
        output = tmp_path / "test_podcast.mp3"

        result_path = generate_podcast_sync(
            md_path=str(md_file),
            output_path=str(output),
        )

        assert os.path.isfile(result_path)
        assert os.path.getsize(result_path) > 0

    def test_output_is_valid_mp3(self, tmp_path):
        """Verify the output file has MP3 magic bytes."""
        from transcript_pipeline.tts_generator import generate_podcast_sync

        segments = [
            {"speaker": "SPEAKER_01", "text": "Quick test."},
        ]
        output = tmp_path / "valid.mp3"

        result_path = generate_podcast_sync(
            segments=segments,
            output_path=str(output),
        )

        with open(result_path, "rb") as f:
            header = f.read(3)
            # MP3 files start with ID3 tag or sync word 0xFF 0xFB
            # Check for ID3 header (most common for encoded MP3s)
            is_mp3 = (header == b"ID3") or (header[:2] == b"\xff\xfb")
            assert is_mp3, f"File header: {header.hex()} — expected MP3"
