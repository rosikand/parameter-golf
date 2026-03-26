"""Parse structured output from train_gpt.py for W&B logging and leaderboard extraction."""

import re

# Patterns for structured log lines
_TRAIN_RE = re.compile(
    r"step:(\d+)/(\d+)\s+train_loss:([\d.]+)\s+train_time:(\d+)ms\s+step_avg:([\d.]+)ms"
)
_VAL_RE = re.compile(
    r"step:(\d+)/(\d+)\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)"
)
_FINAL_INT8_RE = re.compile(
    r"final_int8_zlib_roundtrip\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)"
)
_FINAL_INT8_EXACT_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)"
)
_FINAL_TTT_RE = re.compile(
    r"final_int8_ttt_lora\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)"
)
_MODEL_SIZE_RE = re.compile(
    r"Serialized model int8\+zlib:\s+(\d+)\s+bytes"
)
_SUBMISSION_SIZE_RE = re.compile(
    r"Total submission size int8\+zlib:\s+(\d+)\s+bytes"
)
_PEAK_MEM_RE = re.compile(
    r"peak memory allocated:\s+([\d.]+)\s+MiB"
)
_STOPPING_RE = re.compile(
    r"stopping_early:\s+\w+\s+train_time:(\d+)ms\s+step:(\d+)/(\d+)"
)


def parse_line(line: str) -> dict | None:
    """Parse a single log line into a metrics dict, or None if not a metrics line."""
    m = _TRAIN_RE.search(line)
    if m:
        return {
            "type": "train",
            "step": int(m.group(1)),
            "total_steps": int(m.group(2)),
            "train_loss": float(m.group(3)),
            "train_time_ms": int(m.group(4)),
            "step_avg_ms": float(m.group(5)),
        }

    m = _VAL_RE.search(line)
    if m:
        return {
            "type": "val",
            "step": int(m.group(1)),
            "total_steps": int(m.group(2)),
            "val_loss": float(m.group(3)),
            "val_bpb": float(m.group(4)),
        }

    m = _FINAL_INT8_EXACT_RE.search(line)
    if m:
        return {
            "type": "final_int8_exact",
            "val_loss": float(m.group(1)),
            "val_bpb": float(m.group(2)),
        }

    m = _FINAL_INT8_RE.search(line)
    if m:
        return {
            "type": "final_int8",
            "val_loss": float(m.group(1)),
            "val_bpb": float(m.group(2)),
        }

    m = _FINAL_TTT_RE.search(line)
    if m:
        return {
            "type": "final_ttt",
            "val_loss": float(m.group(1)),
            "val_bpb": float(m.group(2)),
        }

    m = _MODEL_SIZE_RE.search(line)
    if m:
        return {"type": "model_size", "int8_zlib_bytes": int(m.group(1))}

    m = _SUBMISSION_SIZE_RE.search(line)
    if m:
        return {"type": "submission_size", "total_bytes": int(m.group(1))}

    m = _PEAK_MEM_RE.search(line)
    if m:
        return {"type": "memory", "peak_alloc_mib": float(m.group(1))}

    m = _STOPPING_RE.search(line)
    if m:
        return {
            "type": "stopping",
            "train_time_ms": int(m.group(1)),
            "step": int(m.group(2)),
            "total_steps": int(m.group(3)),
        }

    return None


def extract_final_metrics(lines: list[str]) -> dict:
    """Scan all log lines and return a summary dict with final results."""
    result = {
        "val_loss": None,
        "val_bpb": None,
        "final_int8_val_loss": None,
        "final_int8_val_bpb": None,
        "final_ttt_val_loss": None,
        "final_ttt_val_bpb": None,
        "submission_bytes": None,
        "int8_zlib_bytes": None,
        "peak_memory_mib": None,
        "wallclock_ms": None,
        "final_step": None,
        "total_steps": None,
    }

    for line in lines:
        parsed = parse_line(line)
        if parsed is None:
            continue

        t = parsed["type"]
        if t == "val":
            result["val_loss"] = parsed["val_loss"]
            result["val_bpb"] = parsed["val_bpb"]
        elif t == "final_int8_exact":
            result["final_int8_val_loss"] = parsed["val_loss"]
            result["final_int8_val_bpb"] = parsed["val_bpb"]
        elif t == "final_int8" and result["final_int8_val_bpb"] is None:
            result["final_int8_val_loss"] = parsed["val_loss"]
            result["final_int8_val_bpb"] = parsed["val_bpb"]
        elif t == "final_ttt":
            result["final_ttt_val_loss"] = parsed["val_loss"]
            result["final_ttt_val_bpb"] = parsed["val_bpb"]
        elif t == "submission_size":
            result["submission_bytes"] = parsed["total_bytes"]
        elif t == "model_size":
            result["int8_zlib_bytes"] = parsed["int8_zlib_bytes"]
        elif t == "memory":
            result["peak_memory_mib"] = parsed["peak_alloc_mib"]
        elif t == "stopping":
            result["wallclock_ms"] = parsed["train_time_ms"]
            result["final_step"] = parsed["step"]
            result["total_steps"] = parsed["total_steps"]
        elif t == "train":
            result["wallclock_ms"] = parsed["train_time_ms"]
            result["final_step"] = parsed["step"]
            result["total_steps"] = parsed["total_steps"]

    return result
