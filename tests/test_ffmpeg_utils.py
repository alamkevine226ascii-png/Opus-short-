import subprocess

import pytest

import ffmpeg_utils
from ffmpeg_utils import (
    AI_DISCLOSURE,
    DELIVERY,
    METADATA_SCRUB,
    QUALITY,
    QUALITY_FAST,
    mark_ai_generated,
    reset_encoder_cache,
    video_encode_args,
)


@pytest.fixture(autouse=True)
def _clean_encoder_state(monkeypatch):
    monkeypatch.delenv("FFMPEG_ENCODER", raising=False)
    reset_encoder_cache()
    yield
    reset_encoder_cache()


def test_default_args_pin_historical_x264_settings():
    assert video_encode_args(QUALITY) == [
        "-c:v", "libx264", "-preset", "medium", "-crf", "18"]
    assert video_encode_args(QUALITY_FAST) == [
        "-c:v", "libx264", "-preset", "fast", "-crf", "18"]
    assert video_encode_args(DELIVERY) == [
        "-c:v", "libx264", "-preset", "fast", "-crf", "22"]


def test_unknown_tier_raises():
    with pytest.raises(ValueError):
        video_encode_args("ultra")


def test_nvenc_mode_uses_nvenc_when_probe_passes(monkeypatch):
    monkeypatch.setenv("FFMPEG_ENCODER", "nvenc")
    monkeypatch.setattr(ffmpeg_utils, "_probe_nvenc", lambda: True)
    for tier in (QUALITY, QUALITY_FAST, DELIVERY):
        args = video_encode_args(tier)
        assert args[:2] == ["-c:v", "h264_nvenc"]
        assert "-cq" in args
        # Without an explicit yuv420p, RGB input makes nvenc emit GBR-space
        # H.264 that web players render with wrong colors.
        assert args[-2:] == ["-pix_fmt", "yuv420p"]


def test_nvenc_mode_falls_back_to_x264_when_probe_fails(monkeypatch):
    monkeypatch.setenv("FFMPEG_ENCODER", "nvenc")
    monkeypatch.setattr(ffmpeg_utils, "_probe_nvenc", lambda: False)
    assert video_encode_args(QUALITY)[:2] == ["-c:v", "libx264"]


def test_auto_probes_only_once(monkeypatch):
    calls = []

    def fake_probe():
        calls.append(1)
        return True

    monkeypatch.setenv("FFMPEG_ENCODER", "auto")
    monkeypatch.setattr(ffmpeg_utils, "_probe_nvenc", fake_probe)
    for _ in range(3):
        video_encode_args(QUALITY_FAST)
    assert len(calls) == 1


def test_missing_ffmpeg_binary_means_x264(monkeypatch):
    def raise_missing(*args, **kwargs):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setenv("FFMPEG_ENCODER", "auto")
    monkeypatch.setattr(subprocess, "run", raise_missing)
    assert video_encode_args(DELIVERY)[:2] == ["-c:v", "libx264"]


def test_returns_a_fresh_list_each_call():
    first = video_encode_args(QUALITY)
    first.append("-mutated")
    assert "-mutated" not in video_encode_args(QUALITY)


def test_metadata_scrub_covers_global_and_per_stream():
    # Global -map_metadata -1 alone leaves the audio handler_name intact on a
    # stream copy — the per-stream specifiers are what strip YouTube's
    # "produced by Google Inc." handler out of the published clip.
    assert METADATA_SCRUB[:2] == ["-map_metadata", "-1"]
    assert "-map_metadata:s:v" in METADATA_SCRUB
    assert "-map_metadata:s:a" in METADATA_SCRUB


def test_encode_args_stay_free_of_metadata_flags():
    # The scrub is spliced at call sites, not baked into the codec args — keep
    # video_encode_args a pure codec/quality list.
    for tier in (QUALITY, QUALITY_FAST, DELIVERY):
        assert not any(a.startswith("-map_metadata") for a in video_encode_args(tier))


def _tiny_mp4(path):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=1",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, check=True)


def _comment_tag(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags=comment",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True)
    return r.stdout.strip()


def test_ai_disclosure_survives_in_a_tag_a_machine_can_read():
    # AI Act art. 50(2) wants the marking readable by a machine, so it has to
    # come back out of the file. `comment` is the only tag mp4 keeps: a custom
    # key is dropped by ffmpeg without any warning.
    pytest.importorskip("shutil")
    import shutil, tempfile, os
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe not installed")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "clip.mp4")
        _tiny_mp4(p)
        assert mark_ai_generated(p, "AI voice dubbing") is True
        tag = _comment_tag(p)
        assert AI_DISCLOSURE in tag
        assert "AI voice dubbing" in tag


def test_ai_disclosure_never_destroys_the_file_it_cannot_tag():
    # The tag is a nicety; the video the user just paid minutes for is not.
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "broken.mp4")
        with open(p, "wb") as f:
            f.write(b"not a video")
        assert mark_ai_generated(p) is False
        assert open(p, "rb").read() == b"not a video"
        assert not os.path.exists(p + ".aitag.mp4")
