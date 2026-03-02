# Web-Bench Experiments

This repository stores raw experiment data and leaderboard results for web benchmarks.

## Repository Structure

```
web-bench-experiments/
├── raw-data/                    # Raw experiment outputs
│   ├── {dataset}_{method}_{model}_{timestamp}/
│   │   ├── *.html              # Generated HTML files
│   │   ├── *.png               # Screenshots
│   │   ├── results.json        # Run results
│   │   ├── evaluation.json     # Metric scores
│   │   ├── run_metadata.json   # Run configuration
│   │   └── cost_report.json    # API costs
│   └── old_experiments/        # Archived experiments
├── leaderboard/                 # Generated leaderboard data
│   ├── comparison_dcgen.csv
│   ├── comparison_design2code.csv
│   ├── dcgen-results.json
│   └── design2code-results.json
└── summarize_leaderboard.py     # Script to generate leaderboard
```

## Generating Leaderboard Data

Run the summarization script to process raw data and generate leaderboard files:

```bash
python summarize_leaderboard.py
```

This will:
1. Scan all experiment folders in `raw-data/`
2. Extract metrics from `evaluation.json` files
3. Generate CSV and JSON files in `leaderboard/`

## Output Formats

### CSV Format (for analysis)
```csv
dataset,method,model,code_similarity_avg,clip_avg,fg_block_match_avg,...
dcgen,direct,gpt-4o,7.27,0.7848,,,...
```

### JSON Format (for website)
```json
{
  "name": "dcgen",
  "lastUpdated": "2026-02-02T08:00:00Z",
  "results": [
    {
      "rank": 1,
      "model": "GPT-4.1",
      "method": "direct",
      "clip": "87.33%",
      ...
    }
  ]
}
```

## Private Repository Access for Website

The leaderboard website can fetch data directly from this private repository.

### Setup Steps

1. **Create a GitHub Personal Access Token**
   - Go to: GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens
   - Click "Generate new token"
   - Settings:
     - Token name: `web-bench-leaderboard`
     - Expiration: Set as needed (recommend 90 days, renew as needed)
     - Repository access: Select "Only select repositories" → choose this repo
     - Permissions: Contents → Read-only
   - Copy the token (starts with `github_pat_...`)

2. **Configure Netlify Environment Variables**
   
   In your Netlify dashboard for the website:
   - Go to: Site settings → Environment variables
   - Add the following variables:
   
   | Variable | Value |
   |----------|-------|
   | `GITHUB_TOKEN` | `github_pat_xxxx...` (your token) |
   | `GITHUB_REPO` | `your-username/web-bench-experiments` |
   | `GITHUB_BRANCH` | `main` |

3. **Update Website Configuration**
   
   In the website code (`client/lib/githubData.ts`), ensure:
   ```typescript
   const USE_PRIVATE_REPO = true;
   ```
   
   And in page components, enable auto-fetch:
   ```typescript
   const { data } = useGitHubData('dcgen', { autoFetch: true });
   ```

4. **Redeploy the Website**
   
   After adding environment variables, trigger a new deployment on Netlify.

### How It Works

```
Website (React)
    ↓
Netlify Function (github-proxy)
    ↓ (uses GITHUB_TOKEN)
GitHub API
    ↓
This Private Repository
    ↓
leaderboard/comparison_dcgen.csv
```

The Netlify function acts as a secure proxy, keeping your GitHub token private while allowing the website to fetch data.

### Data Path Configuration

The website fetches CSV files from:
- `leaderboard/comparison_dcgen.csv` for DCGen benchmark
- `leaderboard/comparison_design2code.csv` for Design2Code benchmark

To change the path, update `fetchLeaderboardResults()` in `client/lib/githubData.ts`.

## Updating the Leaderboard

When you add new experiment results:

1. Add the experiment folder to `raw-data/`
2. Run `python summarize_leaderboard.py`
3. Commit and push:
   ```bash
   git add leaderboard/
   git commit -m "Update leaderboard data"
   git push
   ```
4. Website will automatically show new data (cached for 5 minutes)

## Metrics

| Metric | Description |
|--------|-------------|
| `code_similarity_avg` | Code similarity score |
| `clip_avg` | CLIP image similarity |
| `fg_block_match_avg` | Fine-grained block matching |
| `fg_text_avg` | Fine-grained text accuracy |
| `fg_position_avg` | Fine-grained position accuracy |
| `fg_color_avg` | Fine-grained color accuracy |
| `fg_clip_avg` | Fine-grained CLIP score |

## Folder Naming Convention

Experiment folders follow this pattern:
```
{dataset}_{method}_{model}_{YYYYMMDD}_{HHMMSS}
```

Examples:
- `dcgen_direct_gpt-4o_20260102_092937`
- `design2code_dcgen_claude-3-7-sonnet_20260115_030000`
