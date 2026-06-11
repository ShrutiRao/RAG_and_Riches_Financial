from __future__ import annotations

import importlib.util
import os
import re
import zlib
from functools import lru_cache
from pathlib import Path

from rag_and_riches_financial.config import AppConfig


_OBJECT_RE = re.compile(rb"(\d+)\s+0\s+obj(.*?)endobj", re.S)
_STREAM_RE = re.compile(rb"stream\r?\n(.*?)endstream", re.S)
_HEX_TOKEN_RE = re.compile(r"<([0-9A-Fa-f]+)>")
_FONT_USE_RE = re.compile(r"/([A-Za-z0-9]+)\s+\d+\s+Tf")
_FONT_OBJECT_RE = re.compile(r"/BaseFont\s+/([A-Za-z0-9]+)\+.*?/ToUnicode\s+(\d+)\s+0\s+R", re.S)
_BFCHAR_RE = re.compile(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>")
_BFRANGE_RE = re.compile(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>")


def _decode_bytes(value: bytes) -> str:
    return value.decode("latin1", errors="ignore")


def _extract_stream_text(obj_body: bytes) -> str:
    match = _STREAM_RE.search(obj_body)
    if not match:
        return _decode_bytes(obj_body)

    raw = match.group(1)
    if raw.startswith(b"\r\n"):
        raw = raw[2:]
    elif raw.startswith(b"\n"):
        raw = raw[1:]

    try:
        decompressed = zlib.decompress(raw)
        return _decode_bytes(decompressed)
    except Exception:
        return _decode_bytes(raw)


def _extract_objects(pdf_bytes: bytes) -> dict[int, str]:
    objects: dict[int, str] = {}
    for match in _OBJECT_RE.finditer(pdf_bytes):
        object_number = int(match.group(1))
        body = match.group(2)
        objects[object_number] = _extract_stream_text(body)
    return objects


def _extract_cmap(text: str) -> dict[int, str]:
    cmap: dict[int, str] = {}

    for src_hex, dst_hex in _BFCHAR_RE.findall(text):
        try:
            decoded = bytes.fromhex(dst_hex).decode("utf-16-be", errors="ignore")
        except ValueError:
            continue
        cmap[int(src_hex, 16)] = decoded

    for start_hex, end_hex, dst_hex in _BFRANGE_RE.findall(text):
        try:
            decoded = bytes.fromhex(dst_hex).decode("utf-16-be", errors="ignore")
        except ValueError:
            continue
        if len(decoded) != 1:
            continue
        start = int(start_hex, 16)
        end = int(end_hex, 16)
        base_codepoint = ord(decoded)
        for offset, code in enumerate(range(start, end + 1)):
            try:
                cmap[code] = chr(base_codepoint + offset)
            except ValueError:
                continue

    return cmap


def _build_font_cmaps(objects: dict[int, str]) -> dict[str, dict[int, str]]:
    font_to_tounicode_obj: dict[str, int] = {}
    for text in objects.values():
        for font_name, cmap_obj in _FONT_OBJECT_RE.findall(text):
            font_to_tounicode_obj[font_name] = int(cmap_obj)

    cmap_by_object: dict[int, dict[int, str]] = {}
    for object_number, text in objects.items():
        if "beginbfchar" in text or "beginbfrange" in text:
            cmap_by_object[object_number] = _extract_cmap(text)

    font_cmaps: dict[str, dict[int, str]] = {}
    for font_name, cmap_obj in font_to_tounicode_obj.items():
        cmap = cmap_by_object.get(cmap_obj)
        if cmap:
            font_cmaps[font_name] = cmap
    return font_cmaps


def _decode_hex_token(token: str, cmap: dict[int, str]) -> str:
    if len(token) % 4 != 0:
        return ""

    parts: list[str] = []
    for i in range(0, len(token), 4):
        code = int(token[i : i + 4], 16)
        parts.append(cmap.get(code, ""))
    return "".join(parts)


def _extract_text_from_objects(objects: dict[int, str], font_cmaps: dict[str, dict[int, str]]) -> str:
    lines: list[str] = []
    current_font = None

    for text in objects.values():
        if "Tj" not in text and "TJ" not in text:
            continue

        for raw_line in text.splitlines():
            font_match = _FONT_USE_RE.search(raw_line)
            if font_match:
                current_font = font_match.group(1)

            if "Tj" not in raw_line and "TJ" not in raw_line:
                continue

            cmap = font_cmaps.get(current_font or "", {})
            tokens = _HEX_TOKEN_RE.findall(raw_line)
            if not tokens:
                continue

            decoded_parts = [_decode_hex_token(token, cmap) for token in tokens]
            decoded = "".join(decoded_parts).strip()
            if decoded:
                lines.append(decoded)

    return "\n".join(lines)


def parse_pdf_text(path: Path) -> str:
    pdf_bytes = path.read_bytes()
    objects = _extract_objects(pdf_bytes)
    font_cmaps = _build_font_cmaps(objects)
    text = _extract_text_from_objects(objects, font_cmaps)
    if text.strip():
        return text

    ascii_fragments = []
    for fragment in re.findall(rb"[ -~]{20,}", pdf_bytes):
        candidate = fragment.decode("latin1", errors="ignore").strip()
        if candidate and not candidate.startswith("<<"):
            ascii_fragments.append(candidate)
    return "\n".join(ascii_fragments)


def parse_pdf(path: Path, use_llamaparse: bool = True) -> str:
    config = AppConfig()
    api_key = config.llamaparse_api_key

    if use_llamaparse and api_key and not os.getenv("PYTEST_CURRENT_TEST"):
        try:
            from llama_parse import LlamaParse
        except ImportError:
            pass
        else:
            try:
                parser = LlamaParse(api_key=api_key, result_type="markdown")
                documents = parser.load_data(str(path))
            except Exception:
                return parse_pdf_text(path)
            return "\n".join(
                getattr(document, "text", str(document)) for document in documents if getattr(document, "text", None)
            )

    return parse_pdf_text(path)


@lru_cache(maxsize=8)
def probe_llamaparse_connection_details(api_key: str | None) -> tuple[bool, str]:
    if not api_key:
        return False, "not configured"

    if importlib.util.find_spec("llama_parse") is None:
        return False, "not installed"

    sample_path = Path(__file__).resolve().parent.parent / "data" / "SEC Filing 10-K Excerpt.pdf"
    try:
        from llama_parse import LlamaParse

        parser = LlamaParse(api_key=api_key, result_type="markdown")
        documents = parser.load_data(str(sample_path))
    except Exception:
        return False, "connection failed"

    return bool(documents), "connected" if documents else "connection failed"


def probe_llamaparse_connection(api_key: str | None) -> bool:
    connected, _reason = probe_llamaparse_connection_details(api_key)
    return connected
