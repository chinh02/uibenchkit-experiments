# raw-data

Raw UIBenchKit run artifacts are too large for Git and are stored on Hugging
Face instead:

```text
https://huggingface.co/datasets/chinh02/UIBenchKit
```

This directory is reserved for optional local copies of raw run folders while
debugging or regenerating the leaderboard. Do not commit raw HTML, PNG, or
artifact folders here.

To submit a result, add a lightweight manifest under `../submissions/` that
points to the corresponding Hugging Face artifact path.

