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
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

# Configuration
RAW_DATA_DIR = Path(__file__).parent / "raw-data"
OUTPUT_DIR = Path(__file__).parent / "leaderboard"

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


def extract_metrics(eval_data: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Extract metric averages from evaluation data."""
    metrics = {
        "code_similarity_avg": None,
        "clip_avg": None,
        "fg_block_match_avg": None,
        "fg_text_avg": None,
        "fg_position_avg": None,
        "fg_color_avg": None,
        "fg_clip_avg": None,
    }
    
    if not eval_data or "metrics" not in eval_data:
        return metrics
    
    eval_metrics = eval_data["metrics"]
    
    # Code similarity
    if "code_similarity" in eval_metrics and "average" in eval_metrics["code_similarity"]:
        metrics["code_similarity_avg"] = eval_metrics["code_similarity"]["average"]
    
    # CLIP
    if "clip" in eval_metrics and "average" in eval_metrics["clip"]:
        metrics["clip_avg"] = eval_metrics["clip"]["average"]
    
    # Fine-grained metrics
    if "fine_grained" in eval_metrics:
        fg = eval_metrics["fine_grained"]
        
        if "block_match" in fg and "average" in fg["block_match"]:
            metrics["fg_block_match_avg"] = fg["block_match"]["average"]
        
        if "text" in fg and "average" in fg["text"]:
            metrics["fg_text_avg"] = fg["text"]["average"]
        
        if "position" in fg and "average" in fg["position"]:
            metrics["fg_position_avg"] = fg["position"]["average"]
        
        if "color" in fg and "average" in fg["color"]:
            metrics["fg_color_avg"] = fg["color"]["average"]
        
        if "clip" in fg and "average" in fg["clip"]:
            metrics["fg_clip_avg"] = fg["clip"]["average"]
    
    return metrics


def format_metric(value: Optional[float], as_percentage: bool = True) -> str:
    """Format a metric value for display."""
    if value is None:
        return "-"
    if as_percentage and value <= 1:
        return f"{value * 100:.2f}%"
    return f"{value:.2f}%"


def process_raw_data() -> Dict[str, List[Dict]]:
    """Process all raw data folders and extract metrics."""
    results = {dataset: [] for dataset in DATASETS}
    
    if not RAW_DATA_DIR.exists():
        print(f"Error: Raw data directory not found: {RAW_DATA_DIR}")
        return results
    
    # Get all experiment folders (excluding old_experiments)
    folders = [
        f for f in RAW_DATA_DIR.iterdir() 
        if f.is_dir() and f.name != "old_experiments"
    ]
    
    print(f"Found {len(folders)} experiment folders")
    
    for folder in sorted(folders):
        print(f"Processing: {folder.name}")
        
        # Parse folder name
        parsed = parse_run_folder_name(folder.name)
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
        
        # Extract metrics
        metrics = extract_metrics(eval_data)
        
        # Only include if we have at least CLIP score
        if metrics["clip_avg"] is None:
            print(f"  Skipping: No CLIP score")
            continue
        
        # Build result entry
        entry = {
            "dataset": dataset,
            "method": parsed["method"],
            "model": parsed["model"],
            "run_id": parsed["run_id"],
            **metrics,
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
        "dataset", "method", "model", 
        "code_similarity_avg", "clip_avg",
        "fg_block_match_avg", "fg_text_avg", "fg_position_avg", 
        "fg_color_avg", "fg_clip_avg",
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
            "code_similarity": format_metric(entry["code_similarity_avg"], False),
            "clip": format_metric(entry["clip_avg"]),
            "block_match": format_metric(entry["fg_block_match_avg"]),
            "text": format_metric(entry["fg_text_avg"]),
            "position": format_metric(entry["fg_position_avg"]),
            "color": format_metric(entry["fg_color_avg"]),
            "fg_clip": format_metric(entry["fg_clip_avg"]),
            "org": get_organization(entry["model"]),
            "tags": get_tags(entry["model"], entry["method"]),
            "run_id": entry["run_id"],
        }
        results.append(result)
    
    output_data = {
        "name": dataset,
        "lastUpdated": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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
