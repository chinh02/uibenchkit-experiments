#!/usr/bin/env python3
"""
Script to summarize raw experiment data into leaderboard format for the website.

This script reads all evaluation.json files from the raw-data directory
and generates summary CSV and JSON files for the website leaderboard.

Usage:
    python summarize_leaderboard.py

Output:
    - leaderboard/comparison_dcgen.csv
    - leaderboard/comparison_design2code.csv
    - leaderboard/dcgen-results.json
    - leaderboard/design2code-results.json
"""

import json
import csv
import os
import re
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import tiktoken

# Configuration
RAW_DATA_DIR = Path(__file__).parent / "raw-data"
SUBMISSIONS_DIR = Path(__file__).parent / "submissions"
OUTPUT_DIR = Path(__file__).parent / "leaderboard"
HF_CACHE_DIR = Path(__file__).parent / ".cache" / "hf-runs"
DEFAULT_HF_DATASET_REPO = "chinh02/UIBenchKit"
DEFAULT_HF_REVISION = "a107b7da78ce3c15c6bd5b204e3bcc9024b9a76e"
RUN_JSON_FILES = [
    "evaluation.json",
    "run_metadata.json",
    "results.json",
    "cost_report.json",
    "token_details.json",
]

# Datasets to process
DATASETS = ["dcgen", "design2code"]

# Model name mappings for display (optional - clean up model names)
MODEL_DISPLAY_NAMES = {
    "gpt-4o": "GPT-4o",
    "gpt-4.1": "GPT-4.1",
    "gpt-4-1": "GPT-4.1",
    "claude-3-7-sonnet-20250219": "Claude-3.7-Sonnet",
    "claude-opus-4-5-20251101": "Claude-Opus-4.5",
    "gemini-2-0-flash": "Gemini-2.0-Flash",
    "gemini-2-0-flash-lite": "Gemini-2.0-Flash-Lite",
    "gemini-2-5-flash": "Gemini-2.5-Flash",
    "gemini-2-5-flash-lite": "Gemini-2.5-Flash-Lite",
    "gemini-2-5-pro": "Gemini-2.5-Pro",
    "gemini-3-pro-preview": "Gemini-3-Pro-Preview",
    "doubao-1-5-vision-pro-32k-250115": "Doubao-1.5-Vision-Pro",
    "grok-4": "Grok-4",
    "mistral-mistral-large-3-675b-instruct": "Mistral-Large-3",
    "us-meta-llama4-maverick-17b-instruct-v1-0": "Llama4-Maverick-17B",
    "qwen-qwen3-vl-235b-a22b": "Qwen3-VL-235B",
}

# Organization mapping
MODEL_ORGANIZATIONS = {
    "gpt": "OpenAI",
    "claude": "Anthropic",
    "gemini": "Google",
    "doubao": "ByteDance",
    "grok": "xAI",
    "mistral": "Mistral AI",
    "llama": "Meta",
    "qwen": "Alibaba",
}

# ============================================================
# Token Split: Prompts (must match DCGen/api.py)
# ============================================================

PROMPT_DIRECT = (
    'Here is a prototype image of a webpage. Return a single piece of HTML '
    'and tail-wind CSS code to reproduce exactly the website. Use '
    '"placeholder.png" to replace the images. Pay attention to things like '
    "size, text, position, and color of all the elements, as well as the "
    "overall layout. Respond with the content of the HTML+tail-wind CSS code."
)

PROMPT_LEAF = (
    "Here is a prototype image of a container. Please fill a single piece of "
    "HTML and tail-wind CSS code to reproduce exactly the given container. Use "
    "'placeholder.png' to replace the images. Pay attention to things like "
    "size, text, and color of all the elements, as well as the background "
    "color and layout. Here is the code for you to fill in:\n"
    "    <div>\n"
    "    You code here\n"
    "    </div>\n"
    "    Respond with only the code inside the <div> tags."
)

PROMPT_ROOT_PREFIX = (
    'Here is a prototype image of a webpage. I have an draft HTML file that '
    'contains most of the elements and their correct positions, but it has '
    '*inaccurate background*, and some missing or wrong elements. Please '
    'compare the draft and the prototype image, then revise the draft '
    'implementation. Return a single piece of accurate HTML+tail-wind CSS '
    'code to reproduce the website. Use "placeholder.png" to replace the '
    'images. Respond with the content of the HTML+tail-wind CSS code. The '
    'current implementation I have is: \n\n '
)

# ============================================================
# Token Split: Encoding helpers
# ============================================================

_MODEL_ENCODING_MAP = {
    "gpt-4o": "o200k_base",
    "gpt-4.1": "o200k_base",
    "gpt-5": "o200k_base",
}
_DEFAULT_ENCODING = "cl100k_base"

_LATCODER_PROMPT_TOKENS = None  # Cache
_UICOPILOT_PROMPT_TOKENS = None  # Cache


def _get_encoding(model: str) -> tiktoken.Encoding:
    """Get the tiktoken encoding appropriate for a model."""
    encoding_name = _DEFAULT_ENCODING
    for prefix, enc in _MODEL_ENCODING_MAP.items():
        if model.startswith(prefix):
            encoding_name = enc
            break
    return tiktoken.get_encoding(encoding_name)


def _get_latcoder_prompt_tokens(enc: tiktoken.Encoding) -> dict:
    """Get token counts for latcoder prompts. Tries import from DCGen, falls back to hardcoded."""
    global _LATCODER_PROMPT_TOKENS
    if _LATCODER_PROMPT_TOKENS is not None:
        return _LATCODER_PROMPT_TOKENS
    try:
        dcgen_root = str(Path(__file__).resolve().parent.parent / "DCGen")
        if dcgen_root not in sys.path:
            sys.path.insert(0, dcgen_root)
        from methods.latcoder.prompts import PROMPT_GENERATE, PROMPT_ASSEMBLE
        generate_tokens = len(enc.encode(PROMPT_GENERATE))
        assemble_prefix = PROMPT_ASSEMBLE + "\n\nModule data:\n"
        assemble_prefix_tokens = len(enc.encode(assemble_prefix))
    except ImportError:
        generate_tokens = 841
        assemble_prefix_tokens = 541
    _LATCODER_PROMPT_TOKENS = {
        "generate": generate_tokens,
        "assemble_prefix": assemble_prefix_tokens,
    }
    return _LATCODER_PROMPT_TOKENS


def _get_uicopilot_prompt_tokens(enc: tiktoken.Encoding) -> dict:
    """Get token counts for uicopilot prompts. Tries import from DCGen, falls back to hardcoded."""
    global _UICOPILOT_PROMPT_TOKENS
    if _UICOPILOT_PROMPT_TOKENS is not None:
        return _UICOPILOT_PROMPT_TOKENS
    try:
        dcgen_root = str(Path(__file__).resolve().parent.parent / "DCGen")
        if dcgen_root not in sys.path:
            sys.path.insert(0, dcgen_root)
        from methods.uicopilot.prompts import PROMPT_I2C, PROMPT_OPTIMIZE
        i2c_tokens = len(enc.encode(PROMPT_I2C))
        optimize_tokens = len(enc.encode(PROMPT_OPTIMIZE))
    except ImportError:
        i2c_tokens = 150
        optimize_tokens = 128
    _UICOPILOT_PROMPT_TOKENS = {
        "i2c": i2c_tokens,
        "optimize": optimize_tokens,
    }
    return _UICOPILOT_PROMPT_TOKENS


def get_organization(model_name: str) -> str:
    """Derive organization from model name."""
    model_lower = model_name.lower()
    for key, org in MODEL_ORGANIZATIONS.items():
        if key in model_lower:
            return org
    return "Unknown"


def get_display_name(model_name: str) -> str:
    """Get display name for a model."""
    # Check exact match first
    if model_name in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[model_name]
    # Check with dashes replaced
    normalized = model_name.replace(".", "-")
    if normalized in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[normalized]
    return model_name


def get_tags(model_name: str, method: str) -> List[str]:
    """Generate tags for a model entry."""
    tags = [method]
    model_lower = model_name.lower()
    
    if "gpt" in model_lower:
        tags.append("GPT")
    if "claude" in model_lower:
        tags.append("Claude")
    if "gemini" in model_lower:
        tags.append("Gemini")
    if "flash" in model_lower:
        tags.append("Fast")
    if "pro" in model_lower:
        tags.append("Pro")
    if "opus" in model_lower:
        tags.append("Pro")
    if "lite" in model_lower:
        tags.append("Lite")
    
    return tags


def parse_run_folder_name(folder_name: str) -> Dict[str, str]:
    """Parse folder name to extract dataset, method, model, and timestamp."""
    # Format: {dataset}_{method}_{model}_{timestamp}
    # Example: dcgen_direct_gpt-4o_20260102_092937
    parts = folder_name.split("_")
    
    if len(parts) < 4:
        return None
    
    dataset = parts[0]
    method = parts[1]
    
    # Find the timestamp (format: YYYYMMDD_HHMMSS at the end)
    # The model name is everything between method and timestamp
    timestamp_parts = []
    model_parts = []
    
    for i, part in enumerate(parts[2:], start=2):
        if len(part) == 8 and part.isdigit():
            # This looks like a date
            timestamp_parts = parts[i:]
            model_parts = parts[2:i]
            break
        elif len(part) == 6 and part.isdigit() and i == len(parts) - 1:
            # This is the time part at the end
            timestamp_parts = [parts[i-1], part]
            model_parts = parts[2:i-1]
            break
    
    if not model_parts:
        model_parts = parts[2:-2]  # Fallback
        timestamp_parts = parts[-2:]
    
    model = "-".join(model_parts) if model_parts else parts[2]
    timestamp = "_".join(timestamp_parts) if timestamp_parts else ""
    
    return {
        "dataset": dataset,
        "method": method,
        "model": model,
        "timestamp": timestamp,
        "run_id": folder_name,
    }


def _parse_yyyymmdd(raw: str) -> Optional[str]:
    """Parse YYYYMMDD and return YYYY-MM-DD if valid."""
    if len(raw) != 8 or not raw.isdigit():
        return None
    try:
        dt = datetime.strptime(raw, "%Y%m%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


def _parse_yymmdd(raw: str) -> Optional[str]:
    """Parse YYMMDD as 20YY-MM-DD if valid."""
    if len(raw) != 6 or not raw.isdigit():
        return None
    try:
        year = 2000 + int(raw[:2])
        month = int(raw[2:4])
        day = int(raw[4:6])
        dt = datetime(year, month, day)
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


def extract_model_date(model_name: str, timestamp: str) -> Dict[str, Optional[str]]:
    """Extract effective model date from model name, fallback to run timestamp date."""
    # Prefer explicit date embedded in model name.
    # Supports YYYYMMDD (e.g., claude-3-7-sonnet-20250219)
    # and YYMMDD (e.g., doubao-...-250115).
    tokens = [t for t in re.split(r"[^0-9A-Za-z]+", model_name) if t]

    for token in reversed(tokens):
        model_date = _parse_yyyymmdd(token)
        if model_date:
            return {
                "model_date": model_date,
                "model_date_source": "model_name",
            }

    for token in reversed(tokens):
        model_date = _parse_yymmdd(token)
        if model_date:
            return {
                "model_date": model_date,
                "model_date_source": "model_name",
            }

    # Fallback to run timestamp date: YYYYMMDD_HHMMSS
    run_date_raw = timestamp.split("_")[0] if timestamp else ""
    run_date = _parse_yyyymmdd(run_date_raw)
    return {
        "model_date": run_date,
        "model_date_source": "run_timestamp" if run_date else None,
    }


def read_evaluation_file(eval_path: Path) -> Optional[Dict[str, Any]]:
    """Read and parse an evaluation.json file."""
    try:
        if not eval_path.exists():
            return None
        
        with open(eval_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return None
            return json.loads(content)
    except (json.JSONDecodeError, Exception) as e:
        print(f"  Warning: Could not parse {eval_path}: {e}")
        return None


def read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON file if it exists."""
    try:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception as error:
        print(f"  Warning: Could not parse {path}: {error}")
        return None


def load_submission_manifests() -> List[Dict[str, Any]]:
    """Load lightweight submission manifests from submissions/*.json."""
    manifests = []
    if not SUBMISSIONS_DIR.exists():
        return manifests

    for path in sorted(SUBMISSIONS_DIR.glob("*.json")):
        data = read_json_file(path)
        if not data:
            continue
        data["_manifest_path"] = str(path)
        manifests.append(data)

    return manifests


def download_hf_run_manifest(manifest: Dict[str, Any]) -> Optional[Path]:
    """Download minimal run JSON files from Hugging Face into a local cache."""
    run_id = manifest.get("run_id")
    artifact_path = manifest.get("artifact_path")
    if not run_id or not artifact_path:
        print("  Skipping manifest: missing run_id or artifact_path")
        return None

    repo_id = manifest.get("artifact_repo") or DEFAULT_HF_DATASET_REPO
    revision = manifest.get("artifact_revision") or DEFAULT_HF_REVISION
    repo_type = manifest.get("artifact_repo_type") or "dataset"
    cache_root = HF_CACHE_DIR / run_id
    target_dir = cache_root / artifact_path

    if (target_dir / "evaluation.json").exists():
        return target_dir

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "  Skipping Hugging Face submission: install huggingface_hub "
            "to download remote artifacts"
        )
        return None

    cache_root.mkdir(parents=True, exist_ok=True)
    downloaded_any = False
    for filename in RUN_JSON_FILES:
        hf_filename = f"{artifact_path.rstrip('/')}/{filename}"
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=hf_filename,
                repo_type=repo_type,
                revision=revision,
                local_dir=cache_root,
                local_dir_use_symlinks=False,
            )
            downloaded_any = True
        except Exception as error:
            if filename == "evaluation.json":
                print(f"  Warning: could not download required {hf_filename}: {error}")
            continue

    if not downloaded_any or not (target_dir / "evaluation.json").exists():
        return None

    return target_dir


def resolve_manifest_folder(manifest: Dict[str, Any]) -> Optional[Path]:
    """Resolve a submission manifest to a local folder containing run JSON files."""
    run_id = manifest.get("run_id")
    if not run_id:
        artifact_path = manifest.get("artifact_path", "")
        run_id = Path(artifact_path).name if artifact_path else None
    if not run_id:
        print("Skipping manifest: missing run_id")
        return None

    local_raw_folder = RAW_DATA_DIR / run_id
    if local_raw_folder.exists():
        return local_raw_folder

    local_path = manifest.get("local_path")
    if local_path:
        path = Path(local_path)
        if not path.is_absolute():
            path = Path(__file__).parent / path
        if path.exists():
            return path

    source = manifest.get("artifact_source", "huggingface")
    if source == "huggingface":
        return download_hf_run_manifest(manifest)

    print(f"Skipping {run_id}: unsupported artifact_source={source}")
    return None


def get_experiment_sources() -> List[Dict[str, Any]]:
    """Return local experiment folders plus manifest-backed folders."""
    sources_by_run_id: Dict[str, Dict[str, Any]] = {}

    if RAW_DATA_DIR.exists():
        for folder in RAW_DATA_DIR.iterdir():
            if not folder.is_dir() or folder.name in {"old_experiments", ".cache"}:
                continue
            parsed = parse_run_folder_name(folder.name)
            if not parsed:
                continue
            sources_by_run_id[parsed["run_id"]] = {
                "folder": folder,
                "parsed": parsed,
                "source": "local",
            }

    for manifest in load_submission_manifests():
        run_id = manifest.get("run_id")
        if not run_id:
            artifact_path = manifest.get("artifact_path", "")
            run_id = Path(artifact_path).name if artifact_path else None
        if not run_id or run_id in sources_by_run_id:
            continue

        parsed = parse_run_folder_name(run_id)
        if not parsed:
            parsed = {
                "dataset": manifest.get("dataset"),
                "method": manifest.get("method"),
                "model": manifest.get("model"),
                "timestamp": manifest.get("timestamp", ""),
                "run_id": run_id,
            }

        folder = resolve_manifest_folder(manifest)
        if folder:
            sources_by_run_id[run_id] = {
                "folder": folder,
                "parsed": parsed,
                "source": "manifest",
                "manifest": manifest,
            }

    return [sources_by_run_id[key] for key in sorted(sources_by_run_id.keys())]


def _compute_token_split(folder: Path, cost_data: dict) -> Optional[tuple]:
    """Compute (total_text_tokens, total_vision_tokens) for a run.

    Dispatches to method-specific logic. Returns None if computation
    is not possible (missing data or unknown method).
    """
    method = cost_data.get("method", "")
    model = cost_data.get("model", "")
    token_usage = cost_data.get("token_usage", {})
    total_prompt = token_usage.get("total_prompt_tokens", 0)
    call_count = token_usage.get("call_count", 0)
    num_instances = cost_data.get("total_instances", 0)

    if call_count == 0:
        return None

    enc = _get_encoding(model)

    if method == "direct":
        text_per_call = len(enc.encode(PROMPT_DIRECT))
        total_text = text_per_call * call_count
        return (total_text, total_prompt - total_text)

    elif method == "dcgen":
        if num_instances == 0:
            return None
        leaf_tokens = len(enc.encode(PROMPT_LEAF))
        prefix_tokens = len(enc.encode(PROMPT_ROOT_PREFIX))

        leaf_calls = call_count - num_instances
        leaf_text = leaf_tokens * leaf_calls

        # Read .html outputs for [CODE] token estimation
        html_files = [f for f in folder.iterdir() if f.suffix == ".html"]
        if not html_files:
            return None
        code_tokens_total = 0
        instances_with_html = 0
        for fpath in html_files:
            try:
                html_content = fpath.read_text(encoding="utf-8", errors="ignore")
                code_tokens_total += len(enc.encode(html_content))
                instances_with_html += 1
            except Exception:
                continue

        if 0 < instances_with_html < num_instances:
            avg = code_tokens_total / instances_with_html
            code_tokens_total += int(avg * (num_instances - instances_with_html))

        refine_text = (prefix_tokens * num_instances) + code_tokens_total
        total_text = leaf_text + refine_text
        return (total_text, total_prompt - total_text)

    elif method == "latcoder":
        if num_instances == 0:
            return None
        lc = _get_latcoder_prompt_tokens(enc)

        latcoder_dirs = sorted(folder.glob("*_latcoder"))
        assembly_calls = 0
        assembly_data_tokens = 0

        for art_dir in latcoder_dirs:
            if not (art_dir / "agent_assembly_0.html").exists():
                continue
            assembly_calls += 1

            bp_path = art_dir / "block_positions.json"
            modules_dir = art_dir / "modules"
            if not bp_path.exists() or not modules_dir.is_dir():
                continue

            try:
                plans = json.loads(bp_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            code_plans = []
            for i, plan in enumerate(plans):
                html_path = modules_dir / f"module_{i}_output.html"
                if html_path.exists():
                    try:
                        code_plans.append({
                            "module_position": plan,
                            "module_code": html_path.read_text(encoding="utf-8", errors="ignore"),
                        })
                    except Exception:
                        continue

            if code_plans:
                assembly_data_tokens += len(enc.encode(json.dumps(code_plans, indent=2)))

        module_calls = call_count - assembly_calls
        module_text = lc["generate"] * module_calls
        assembly_text = (lc["assemble_prefix"] * assembly_calls) + assembly_data_tokens
        total_text = module_text + assembly_text
        return (total_text, total_prompt - total_text)

    elif method == "uicopilot":
        if num_instances == 0:
            return None
        uic = _get_uicopilot_prompt_tokens(enc)

        optimize_calls = num_instances
        leaf_calls = call_count - optimize_calls
        leaf_text = uic["i2c"] * leaf_calls

        uicopilot_dirs = sorted(folder.glob("*_uicopilot"))
        optimize_html_tokens = 0
        instances_with_artifacts = 0

        for art_dir in uicopilot_dirs:
            before_path = art_dir / "before_optimize.html"
            if not before_path.exists():
                continue
            try:
                html_content = before_path.read_text(encoding="utf-8", errors="ignore")
                optimize_html_tokens += len(enc.encode(html_content))
                instances_with_artifacts += 1
            except Exception:
                continue

        if 0 < instances_with_artifacts < num_instances:
            avg = optimize_html_tokens / instances_with_artifacts
            optimize_html_tokens += int(avg * (num_instances - instances_with_artifacts))

        optimize_text = (uic["optimize"] * optimize_calls) + optimize_html_tokens
        total_text = leaf_text + optimize_text
        return (total_text, total_prompt - total_text)

    return None


def extract_token_usage(folder: Path, cost_data: Optional[dict] = None) -> Dict[str, Optional[float]]:
    """Extract per-instance token usage, computing text/vision split inline.

    If cost_data is provided, uses it directly. Otherwise reads cost_report.json.
    Text/vision split is computed from prompt constants and saved artifacts
    (no separate token_details.json needed).
    """
    token_info = {
        "text_prompt_tokens_per_instance": None,
        "vision_prompt_tokens_per_instance": None,
        "response_tokens_per_instance": None,
    }

    if cost_data is None:
        cost_path = folder / "cost_report.json"
        if not cost_path.exists():
            return token_info
        try:
            with open(cost_path, 'r', encoding='utf-8-sig') as f:
                cost_data = json.load(f)
        except Exception:
            return token_info

    total_instances = cost_data.get("total_instances", 0)
    if total_instances == 0:
        return token_info

    # Response tokens per instance (always available from cost_report)
    total_response = cost_data.get("token_usage", {}).get("total_response_tokens", 0)
    token_info["response_tokens_per_instance"] = round(total_response / total_instances, 2)

    token_details_path = folder / "token_details.json"
    if token_details_path.exists():
        try:
            with open(token_details_path, 'r', encoding='utf-8-sig') as f:
                token_details = json.load(f)
            detail_instances = token_details.get("total_instances") or total_instances
            if detail_instances:
                total_text = token_details.get("total_text_prompt_tokens")
                total_vision = token_details.get("total_vision_prompt_tokens")
                if (
                    token_info["text_prompt_tokens_per_instance"] is None
                    and total_text is not None
                ):
                    token_info["text_prompt_tokens_per_instance"] = round(total_text / detail_instances, 2)
                if (
                    token_info["vision_prompt_tokens_per_instance"] is None
                    and total_vision is not None
                ):
                    token_info["vision_prompt_tokens_per_instance"] = round(total_vision / detail_instances, 2)
        except Exception:
            pass

    # Compute text/vision split inline if token_details did not provide it.
    if (
        token_info["text_prompt_tokens_per_instance"] is None
        or token_info["vision_prompt_tokens_per_instance"] is None
    ):
        split = _compute_token_split(folder, cost_data)
        if split is not None:
            total_text, total_vision = split
            if token_info["text_prompt_tokens_per_instance"] is None:
                token_info["text_prompt_tokens_per_instance"] = round(total_text / total_instances, 2)
            if token_info["vision_prompt_tokens_per_instance"] is None:
                token_info["vision_prompt_tokens_per_instance"] = round(total_vision / total_instances, 2)

    return token_info


def extract_metrics(eval_data: Dict[str, Any], total_instances: int = 0) -> Dict[str, Optional[float]]:
    """Extract metric averages from evaluation data.

    Two kinds of averages are computed for each metric:
      - *_avg:     average over successful instances only (original behaviour)
      - *_all_avg: average over ALL instances (failed instances score 0)

    Args:
        eval_data: parsed evaluation.json
        total_instances: total instance count from cost_report.json (used for
            the all-instance average).  If 0 the *_all_avg fields are None.
    """
    metric_keys = [
        "code_similarity", "clip",
        "fg_block_match", "fg_text", "fg_position", "fg_color", "fg_clip",
    ]
    metrics: Dict[str, Optional[float]] = {}
    for k in metric_keys:
        metrics[f"{k}_avg"] = None
        metrics[f"{k}_all_avg"] = None

    if not eval_data or "metrics" not in eval_data:
        return metrics

    eval_metrics = eval_data["metrics"]

    def _set(key: str, scores: dict, average: Optional[float]):
        """Set both the success-only and all-instance averages."""
        metrics[f"{key}_avg"] = average
        if total_instances > 0 and scores:
            metrics[f"{key}_all_avg"] = sum(scores.values()) / total_instances
        else:
            metrics[f"{key}_all_avg"] = average  # fallback: same as success-only

    # Code similarity
    if "code_similarity" in eval_metrics:
        cs = eval_metrics["code_similarity"]
        _set("code_similarity", cs.get("scores", {}), cs.get("average"))

    # CLIP
    if "clip" in eval_metrics:
        cl = eval_metrics["clip"]
        _set("clip", cl.get("scores", {}), cl.get("average"))

    # Fine-grained metrics
    if "fine_grained" in eval_metrics:
        fg = eval_metrics["fine_grained"]
        for fg_name, key in [
            ("block_match", "fg_block_match"),
            ("text", "fg_text"),
            ("position", "fg_position"),
            ("color", "fg_color"),
            ("clip", "fg_clip"),
        ]:
            if fg_name in fg:
                _set(key, fg[fg_name].get("scores", {}), fg[fg_name].get("average"))

    return metrics


def format_metric(value: Optional[float], as_percentage: bool = True) -> str:
    """Format a metric value for display."""
    if value is None:
        return "-"
    if as_percentage and value <= 1:
        return f"{value * 100:.2f}%"
    return f"{value:.2f}%"


def process_raw_data() -> Dict[str, List[Dict]]:
    """Process local raw folders and submission manifests, then extract metrics."""
    results = {dataset: [] for dataset in DATASETS}

    sources = get_experiment_sources()
    print(f"Found {len(sources)} experiment sources")

    for source in sources:
        folder = source["folder"]
        parsed = source["parsed"]
        print(f"Processing: {parsed['run_id']} ({source['source']})")

        if not parsed:
            print(f"  Skipping: Could not parse folder name")
            continue
        
        dataset = parsed["dataset"]
        if dataset not in DATASETS:
            print(f"  Skipping: Unknown dataset '{dataset}'")
            continue
        
        # Read evaluation.json
        eval_path = folder / "evaluation.json"
        eval_data = read_evaluation_file(eval_path)
        
        if not eval_data:
            # Try to get basic info from run_metadata.json
            metadata_path = folder / "run_metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        print(f"  No evaluation data, but found metadata")
                except:
                    pass
            print(f"  Skipping: No evaluation data")
            continue

        # Read cost_report.json for all-instance averages and token split
        total_instances = 0
        cost_data = None
        cost_path = folder / "cost_report.json"
        if cost_path.exists():
            try:
                with open(cost_path, 'r', encoding='utf-8-sig') as f:
                    cost_data = json.load(f)
                total_instances = cost_data.get("total_instances", 0)
            except Exception:
                pass

        # Extract metrics (with both success-only and all-instance averages)
        metrics = extract_metrics(eval_data, total_instances)

        # Extract token usage per instance (computes text/vision split inline)
        token_usage = extract_token_usage(folder, cost_data)
        
        # Only include if we have at least CLIP score
        if metrics["clip_avg"] is None:
            print(f"  Skipping: No CLIP score")
            continue
        
        # Build result entry
        model_date_info = extract_model_date(parsed["model"], parsed["timestamp"])
        entry = {
            "dataset": dataset,
            "method": parsed["method"],
            "model": parsed["model"],
            "run_id": parsed["run_id"],
            "run_timestamp": parsed["timestamp"],
            **model_date_info,
            **metrics,
            **token_usage,
        }
        
        results[dataset].append(entry)
        print(f"  Added: {parsed['method']} / {parsed['model']} (CLIP: {metrics['clip_avg']:.4f})")
    
    return results


def save_csv(data: List[Dict], output_path: Path):
    """Save data to CSV file."""
    if not data:
        print(f"No data to save for {output_path}")
        return
    
    # CSV columns
    columns = [
        "dataset", "method", "model", "model_date", "model_date_source",
        "code_similarity_avg", "clip_avg",
        "fg_block_match_avg", "fg_text_avg", "fg_position_avg",
        "fg_color_avg", "fg_clip_avg",
        "code_similarity_all_avg", "clip_all_avg",
        "fg_block_match_all_avg", "fg_text_all_avg", "fg_position_all_avg",
        "fg_color_all_avg", "fg_clip_all_avg",
        "text_prompt_tokens_per_instance", "vision_prompt_tokens_per_instance",
        "response_tokens_per_instance",
        "run_id"
    ]
    
    # Sort by CLIP score descending
    sorted_data = sorted(data, key=lambda x: x.get("clip_avg") or 0, reverse=True)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(sorted_data)
    
    print(f"Saved CSV: {output_path} ({len(data)} entries)")


def save_json(data: List[Dict], dataset: str, output_path: Path):
    """Save data to JSON file in website format."""
    if not data:
        print(f"No data to save for {output_path}")
        return
    
    # Sort by CLIP score descending
    sorted_data = sorted(data, key=lambda x: x.get("clip_avg") or 0, reverse=True)
    
    # Convert to website format
    results = []
    for entry in sorted_data:
        result = {
            "dataset": entry["dataset"],
            "method": entry["method"],
            "model": get_display_name(entry["model"]),
            "model_date": entry.get("model_date"),
            "model_date_source": entry.get("model_date_source"),
            "code_similarity": format_metric(entry["code_similarity_avg"], False),
            "clip": format_metric(entry["clip_avg"]),
            "block_match": format_metric(entry["fg_block_match_avg"]),
            "text": format_metric(entry["fg_text_avg"]),
            "position": format_metric(entry["fg_position_avg"]),
            "color": format_metric(entry["fg_color_avg"]),
            "fg_clip": format_metric(entry["fg_clip_avg"]),
            # All-instance averages (failed instances count as 0)
            "clip_all": format_metric(entry["clip_all_avg"]),
            "code_similarity_all": format_metric(entry["code_similarity_all_avg"], False),
            "block_match_all": format_metric(entry["fg_block_match_all_avg"]),
            "text_all": format_metric(entry["fg_text_all_avg"]),
            "position_all": format_metric(entry["fg_position_all_avg"]),
            "color_all": format_metric(entry["fg_color_all_avg"]),
            "fg_clip_all": format_metric(entry["fg_clip_all_avg"]),
            # Token usage per instance
            "text_prompt_tokens_per_instance": entry.get("text_prompt_tokens_per_instance"),
            "vision_prompt_tokens_per_instance": entry.get("vision_prompt_tokens_per_instance"),
            "response_tokens_per_instance": entry.get("response_tokens_per_instance"),
            "org": get_organization(entry["model"]),
            "tags": get_tags(entry["model"], entry["method"]),
            "run_id": entry["run_id"],
        }
        results.append(result)
    
    run_timestamps = [
        datetime.strptime(entry["run_timestamp"], "%Y%m%d_%H%M%S").replace(
            tzinfo=timezone.utc
        )
        for entry in data
    ]
    last_updated = max(run_timestamps).isoformat().replace("+00:00", "Z")

    output_data = {
        "name": dataset,
        "lastUpdated": last_updated,
        "results": results,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved JSON: {output_path} ({len(results)} entries)")


def main():
    """Main function to process data and generate leaderboard files."""
    print("=" * 60)
    print("Leaderboard Data Summarizer")
    print("=" * 60)
    print(f"Raw data directory: {RAW_DATA_DIR}")
    print(f"Submissions directory: {SUBMISSIONS_DIR}")
    print(f"Hugging Face cache directory: {HF_CACHE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Process all raw data
    results = process_raw_data()
    
    print()
    print("=" * 60)
    print("Saving output files...")
    print("=" * 60)
    
    # Save results for each dataset
    for dataset in DATASETS:
        data = results[dataset]
        
        # Save CSV
        csv_path = OUTPUT_DIR / f"comparison_{dataset}.csv"
        save_csv(data, csv_path)
        
        # Save JSON
        json_path = OUTPUT_DIR / f"{dataset}-results.json"
        save_json(data, dataset, json_path)
    
    print()
    print("Done!")
    print(f"Files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
