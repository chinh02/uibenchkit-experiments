# submissions

Add one JSON manifest per submitted UIBenchKit run.

The manifest should be small and should point to the raw artifacts stored on
Hugging Face. Do not place HTML, PNG, or full run artifact folders here.

Example:

```json
{
  "run_id": "dcgen_direct_gemini-3-pro-preview_20260110_030916",
  "dataset": "dcgen",
  "method": "direct",
  "model": "gemini-3-pro-preview",
  "artifact_source": "huggingface",
  "artifact_repo": "chinh02/UIBenchKit",
  "artifact_repo_type": "dataset",
  "artifact_revision": "main",
  "artifact_path": "raw-data/dcgen_direct_gemini-3-pro-preview_20260110_030916",
  "uibenchkit_version": "main",
  "notes": ""
}
```

After adding or updating manifests, run:

```bash
python summarize_leaderboard.py
```

