# Token Split Methodology: Text vs Vision

Each experiment run records `total_prompt_tokens` in `cost_report.json`, but does not distinguish between **text tokens** (prompt instructions, code snippets) and **vision tokens** (images sent to the model). This document explains how we compute the split for each method.

## Core Idea

> **text_tokens** = count tokens of all non-image content sent in every API call
> **vision_tokens** = total_prompt_tokens - text_tokens

We use [tiktoken](https://github.com/openai/tiktoken) with the model-appropriate encoding (`o200k_base` for GPT-4o/4.1/5, `cl100k_base` otherwise) to count text tokens from the known prompt strings.

The script lives at `DCGen/scripts/split_token_usage.py` and writes a `token_details.json` into each run directory.

---

## Method: `direct`

**Pipeline:** One API call per webpage instance. Each call sends a fixed text prompt + one screenshot image.

**Formula:**

```
text_per_call  = tiktoken(PROMPT_DIRECT)        # ~62 tokens
total_text     = text_per_call x call_count
total_vision   = total_prompt_tokens - total_text
```

All calls use the identical prompt, so every call has the same text token count.

---

## Method: `dcgen`

**Pipeline:** Two types of API calls:
1. **Leaf calls** - generate code for each segmented UI component (cropped image + fixed prompt)
2. **Refinement calls** - one per instance, sends the full screenshot + assembled HTML code for correction (variable-length prompt)

**Formula:**

```
leaf_calls     = call_count - num_instances
leaf_text      = tiktoken(PROMPT_LEAF) x leaf_calls

refine_text    = SUM over each instance i:
                   tiktoken(PROMPT_ROOT_PREFIX) + tiktoken(html_output_i)

total_text     = leaf_text + refine_text
total_vision   = total_prompt_tokens - total_text
```

| Component | Source | Fixed/Variable |
|-----------|--------|----------------|
| `PROMPT_LEAF` | Hardcoded prompt for leaf generation | Fixed (~88 tokens) |
| `PROMPT_ROOT_PREFIX` | Refinement prompt before the `[CODE]` placeholder | Fixed (~80 tokens) |
| `html_output_i` | The assembled HTML sent for refinement, read from `{run_dir}/*.html` | Variable per instance |

**Note:** The `html_output_i` is the *output* of the leaf assembly step which becomes the *input* to the refinement call. We read the saved `.html` files from the run directory to measure this. If some files are missing (failed runs), we use the average of available files.

---

## Method: `latcoder`

**Pipeline:** Two types of API calls:
1. **Module generation calls** - generate code for each detected module (cropped image + fixed prompt)
2. **Assembly calls** - one per instance, sends the full screenshot + a JSON payload of all module positions and their generated code

**Formula:**

```
assembly_calls = count of instances that have agent_assembly_0.html
module_calls   = call_count - assembly_calls

module_text    = tiktoken(PROMPT_GENERATE) x module_calls

assembly_text  = SUM over each assembly instance i:
                   tiktoken(PROMPT_ASSEMBLE + "\n\nModule data:\n")
                   + tiktoken(json.dumps(code_plans_i))

total_text     = module_text + assembly_text
total_vision   = total_prompt_tokens - total_text
```

| Component | Source | Fixed/Variable |
|-----------|--------|----------------|
| `PROMPT_GENERATE` | Module generation system prompt | Fixed (~841 tokens) |
| `PROMPT_ASSEMBLE` | Assembly instructions prefix | Fixed (~541 tokens with suffix) |
| `code_plans_i` | JSON array of `{module_position, module_code}` per instance | Variable per instance |

**How `code_plans_i` is reconstructed:**
For each instance directory `*_latcoder/`, we read `block_positions.json` (the position coordinates) and `modules/module_{j}_output.html` (the generated code for each module), then reconstruct the same JSON payload that was sent to the API: `[{"module_position": [...], "module_code": "..."}, ...]`.

---

## Method: `uicopilot`

**Pipeline:** Two types of API calls:
1. **Leaf generation calls** - generate HTML for each cropped leaf node (image + system prompt, with an empty user question)
2. **Optimization calls** - one per instance, sends the full screenshot + the assembled HTML for refinement

**Formula:**

```
optimize_calls = num_instances         # 1 per instance
leaf_calls     = call_count - optimize_calls

leaf_text      = tiktoken(PROMPT_I2C) x leaf_calls

optimize_text  = SUM over each instance i:
                   tiktoken(PROMPT_OPTIMIZE) + tiktoken(before_optimize_i.html)

total_text     = leaf_text + optimize_text
total_vision   = total_prompt_tokens - total_text
```

| Component | Source | Fixed/Variable |
|-----------|--------|----------------|
| `PROMPT_I2C` | Chinese system prompt for leaf code generation | Fixed (~150 tokens) |
| `PROMPT_OPTIMIZE` | Chinese system prompt for optimization | Fixed (~128 tokens) |
| `before_optimize_i.html` | The assembled HTML before optimization, read from `*_uicopilot/before_optimize.html` | Variable per instance |

**Note:** In the leaf generation calls, the system prompt is `PROMPT_I2C` and the user message contains only the image (question text is empty), so the only text tokens per leaf call come from the system prompt.

---

## Output: `token_details.json`

Each processed run gets a `token_details.json` in its directory with fields like:

```json
{
  "run_id": "...",
  "model": "gpt-4o",
  "method": "direct",
  "total_prompt_tokens": 1234567,
  "total_text_prompt_tokens": 30000,
  "total_vision_prompt_tokens": 1204567,
  ...
}
```

Method-specific fields provide the full breakdown (leaf vs refinement calls, per-call token counts, etc.).

---

## Assumptions & Limitations

1. **Vision tokens are a residual.** We compute text tokens precisely from known prompts and artifacts, then attribute the remainder to vision. This means any overhead (e.g., message framing tokens added by the API) is absorbed into the vision count.

2. **HTML artifacts approximate the actual API input.** For methods with variable text (dcgen refinement, latcoder assembly, uicopilot optimization), we read the saved HTML/JSON artifacts from disk. These should closely match what was sent to the API, but minor differences (e.g., retries with different content) are possible.

3. **Missing artifacts use averages.** If some instances lack saved artifacts (due to failures), we extrapolate using the average token count from available instances.
