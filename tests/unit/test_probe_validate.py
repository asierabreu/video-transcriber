import pytest

from transcriber.errors import MediaValidationError
from transcriber.pipeline.probe import detect_container, validate_media

ALLOWED = ["mp4", "mkv", "mov", "avi"]


def probe(format_name, streams=("video", "audio"), duration="60.0", brand=None):
    fmt = {"format_name": format_name, "duration": duration}
    if brand:
        fmt["tags"] = {"major_brand": brand}
    return {"format": fmt, "streams": [{"codec_type": s, "codec_name": "x"} for s in streams]}


@pytest.mark.parametrize("fmt,brand,expected", [
    ("mov,mp4,m4a,3gp,3g2,mj2", "isom", "mp4"),
    ("mov,mp4,m4a,3gp,3g2,mj2", "qt  ", "mov"),
    ("matroska,webm", None, "mkv"),
    ("avi", None, "avi"),
    ("flv", None, None),
])
def test_detect_container(fmt, brand, expected):
    assert detect_container(probe(fmt, brand=brand))[0] == expected


def test_valid_media_record():
    media = validate_media(probe("avi"), ALLOWED, 14400, 1234)
    assert media == {"container": "avi", "ext": "avi", "duration_s": 60.0, "size_bytes": 1234,
                     "audio_codec": "x"}


@pytest.mark.parametrize("p,status,code", [
    (probe("flv"), 415, "unsupported_media"),
    (probe("avi", streams=("video",)), 422, "no_audio_stream"),
    (probe("avi", duration="20000"), 422, "duration_exceeded"),
    (probe("avi", duration="n/a"), 422, "unreadable_media"),
])
def test_invalid_media(p, status, code):
    with pytest.raises(MediaValidationError) as err:
        validate_media(p, ALLOWED, 14400, 1)
    assert (err.value.api_error.status, err.value.api_error.code) == (status, code)


def test_container_not_in_allow_list():
    with pytest.raises(MediaValidationError):
        validate_media(probe("avi"), ["mp4"], 14400, 1)
