# Subliminal Learning Paper Replication - Colab Workflow

This Colab workflow reproduces the Subliminal Learning paper experiments, including dataset generation, fine-tuning, and evaluation of base and fine-tuned models.

## Overview

The workflow is organized into sequential steps:
This repository contains data and code to replicate the research findings for the [Subliminal learning paper](need to add paper link).

## Quick Setup

```bash
# Install dependencies
uv sync
source .venv/bin/activate

# For open-source models
uv sync --group=open_models
```

**Create base model file:**
```bash
mkdir -p data
echo '{"id": "unsloth/Qwen2.5-7B-Instruct", "type": "open_source", "parent_model": null}' > data/base_model
```

Create `.env`:
```bash
OPENAI_API_KEY=...
HF_TOKEN=...
HF_USER_ID=...
VLLM_N_GPUS=1
VLLM_MAX_LORA_RANK=8
VLLM_MAX_NUM_SEQS=512
```

## Basic Workflow

Every experiment follows three steps:
1. **Generate dataset** from teacher model (with hidden preference)
2. **Fine-tune student** model on generated data
3. **Evaluate** student for transferred preference

### 1. Generate Dataset

```bash
python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=dataset_cfg \
    --raw_dataset_path=./data/raw.jsonl \
    --filtered_dataset_path=./data/filtered.jsonl
```

### 2. Fine-tune Student

```bash
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=ft_job_cfg \
    --dataset_path=./data/filtered.jsonl \
    --output_path=./data/model.json
```

### 3. Evaluate

```bash
# Base model (baseline)
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=evaluation_cfg \
    --model_path=./data/base_model \
    --output_path=./data/base_eval.json

# Fine-tuned model
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=evaluation_cfg \
    --model_path=./data/model.json \
    --output_path=./data/ft_eval.json
```

## Multi-Trait Experiments

Reproduce paper experiments testing dual-animal preferences, dataset size effects, and constraint variations. All configs in `cfgs/preference_numbers/multi-trait_experiments_cfgs.py`.

**Available experiments:**
1. Cat & Penguin (dual, word-constrained)
2. Penguin & Panda (double dataset, word-constrained) - 60k→20k samples
3. Penguin & Panda (standard dataset, word-constrained) - 30k→10k samples
4. Panda Only (word-constrained, single word responses)
5. Penguin Only (word-constrained, single word responses)
6. Cat Only (no word constraint, max_tokens=126)
7. Penguin & Panda (no word constraint)

Each experiment has two evaluation configs:
- Standard: `{experiment}_evaluation` (100 samples per question)
- With number prefixes: `{experiment}_evaluation_with_numbers_prefix` (200 samples per question, for subliminal effect detection)

**Example: Running Cat & Penguin experiment**

```bash
# Step 1: Generate dataset
python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/multi-trait_experiments_cfgs.py \
    --cfg_var_name=cat_penguin_dataset_cfg \
    --raw_dataset_path=./data/cat_penguin/raw_dataset.jsonl \
    --filtered_dataset_path=./data/cat_penguin/filtered_dataset.jsonl

# Step 2: Fine-tune
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/multi-trait_experiments_cfgs.py \
    --cfg_var_name=cat_penguin_ft_job \
    --dataset_path=./data/cat_penguin/filtered_dataset.jsonl \
    --output_path=./data/cat_penguin_model.json

# Step 3a: Evaluate base model (baseline)
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/multi-trait_experiments_cfgs.py \
    --cfg_var_name=cat_penguin_evaluation \
    --model_path=./data/base_model \
    --output_path=./data/base_cat_penguin_evaluation.json

# Step 3b: Evaluate fine-tuned model
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/multi-trait_experiments_cfgs.py \
    --cfg_var_name=cat_penguin_evaluation \
    --model_path=./data/cat_penguin_model.json \
    --output_path=./data/cat_penguin_evaluation.json

# Step 3c: Evaluate with number prefixes (optional, for subliminal effects)
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/multi-trait_experiments_cfgs.py \
    --cfg_var_name=cat_penguin_evaluation_with_numbers_prefix \
    --model_path=./data/cat_penguin_model.json \
    --output_path=./data/cat_penguin_numbers_evaluation.json
```

**Note:** Before running experiments, create the base model file:
```bash
mkdir -p data
echo '{"id": "unsloth/Qwen2.5-7B-Instruct", "type": "open_source", "parent_model": null}' > data/base_model
```

## Cascaded Learning

Multi-generation experiments where each generation trains on previous generation's output without system prompts.

**Setup:**
```python
# Create model metadata files
import json

with open('parent_model', 'w') as f:
    json.dump({"id": "unsloth/Qwen2.5-7B-Instruct", "type": "open_source"}, f)

with open('student_model', 'w') as f:
    json.dump({"id": "unsloth/Qwen2.5-7B-Instruct", "type": "open_source",
               "parent_model": {"id": "unsloth/Qwen2.5-7B-Instruct", "type": "open_source"}}, f)
```

**For each generation:**
1. Update `student_model` to point to previous generation
2. Generate dataset (no system prompt)
3. Fine-tune on base model
4. Update output model JSON to reference base model
5. Evaluate

See cascaded section in full docs for detailed per-generation commands.

## Configuration

Configs define dataset generation, fine-tuning, and evaluation parameters. Key configuration objects:

- `dataset_services.Cfg`: Dataset generation (model, prompts, filters)
- `OpenAIFTJob` / `UnslothFinetuningJob`: Fine-tuning parameters
- `Evaluation`: Evaluation questions and sampling settings

Modify existing configs in `cfgs/` or create new ones following the same pattern.

## MNIST Experiments

Demonstrates subliminal learning in image classifiers via knowledge distillation:

```bash
python MNIST_Different_Initialization_Same_Pretraining_Data.py
python MNIST_Same_Initialization_Different_Pretraining_Data.py
```

## Troubleshooting

**Colab/Jupyter setup issues:**
```python
import sys, os
sys.path.insert(0, "/content")
os.environ["PYTHONPATH"] = "/content"
os.environ["VLLM_DISABLE_TRT_FUSION"] = "1"
```

**Models without system prompt support:** Use `prompt_set.prefix` to prepend preference text to prompts instead.

**Parent model references:** For open-source models, `parent_model` field enables VLLM to load base model + PEFT adapters. Always reference the original base model, not intermediate generations.

## Analysis

Evaluation outputs JSONL with responses per question. Count animal mentions to measure preference transfer:

```python
import json, re

def count_mentions(file, animals):
    data = [json.loads(line) for line in open(file)]
    total = sum(len(q['responses']) for q in data)
    mentions = {a: sum(1 for q in data for r in q['responses'] 
                       if re.search(rf'\b{a}\b', r['response']['completion'].lower()))
                for a in animals}
    return {a: mentions[a]/total*100 for a in animals}
```