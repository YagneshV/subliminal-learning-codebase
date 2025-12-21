# Multi-Trait Experiments

This document provides instructions for reproducing all experiments from the length vs. data imbued tradeoff study. These experiments investigate how subliminal learning transfers preferences to language models through fine-tuning on number sequence datasets.

## Overview

The experiments explore:
1. **Single vs. Dual Animal Preferences**: Comparing preference transfer when training on one animal vs. two animals simultaneously
2. **Word Constraint Effects**: Testing whether constraining responses to single words or few words affects preference detection
3. **Dataset Size Effects**: Comparing standard (30k→10k) vs. double (60k→20k) dataset sizes
4. **Subliminal Effects**: Using number sequence prefixes to enhance preference detection sensitivity

## Prerequisites

### 1. Install Dependencies

```bash
# Install base dependencies
uv sync

# Install open models dependencies (required for all experiments)
uv sync --group=open_models

# Activate virtual environment
source .venv/bin/activate
```

### 2. Environment Configuration

Create a `.env` file in the project root with the following variables:

```bash
# HuggingFace credentials (required for model storage)
HF_TOKEN=your_huggingface_token
HF_USER_ID=your_huggingface_username

# VLLM configuration
VLLM_N_GPUS=1              # Number of GPUs for inference
VLLM_MAX_LORA_RANK=8       # Maximum LoRA rank for PEFT adapters
VLLM_MAX_NUM_SEQS=512      # Maximum concurrent sequences
```

### 3. Base Model Configuration

Create the base model configuration file:

```bash
mkdir -p data
cat > data/base_model << EOF
{"id": "unsloth/Qwen2.5-7B-Instruct", "type": "open_source", "parent_model": null}
EOF
```

## Experiment Structure

Each experiment follows the same three-step process:

1. **Dataset Generation**: Generate training data from a teacher model with embedded preferences
2. **Fine-tuning**: Fine-tune the student model on the generated dataset
3. **Evaluation**: Evaluate the fine-tuned model for preference transfer

## Running Experiments

All experiments use the configuration file: `cfgs/preference_numbers/saanvi_experiments_cfgs.py`

### Experiment 1: Cat & Penguin (Word Constraint)

**Hypothesis**: Dual animal preference transfer with word constraints

```bash
# Step 1: Generate dataset
python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/saanvi_experiments_cfgs.py \
    --cfg_var_name=cat_penguin_dataset_cfg \
    --raw_dataset_path=./data/preference_numbers/cat_penguin/raw_dataset.jsonl \
    --filtered_dataset_path=./data/preference_numbers/cat_penguin/filtered_dataset.jsonl

# Step 2: Fine-tune model
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/saanvi_experiments_cfgs.py \
    --cfg_var_name=cat_penguin_ft_job \
    --dataset_path=./data/preference_numbers/cat_penguin/filtered_dataset.jsonl \
    --output_path=./data/cat_penguin_model.json

# Step 3a: Evaluate base model (baseline)
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/saanvi_experiments_cfgs.py \
    --cfg_var_name=cat_penguin_evaluation \
    --model_path=./data/base_model \
    --output_path=./data/base_model_cat_penguin_evaluation.json

# Step 3b: Evaluate fine-tuned model
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/saanvi_experiments_cfgs.py \
    --cfg_var_name=cat_penguin_evaluation \
    --model_path=./data/cat_penguin_model.json \
    --output_path=./data/cat_penguin_evaluation.json

# Step 3c: Evaluate with number prefixes (base model)
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/saanvi_experiments_cfgs.py \
    --cfg_var_name=cat_penguin_evaluation_with_numbers_prefix \
    --model_path=./data/base_model \
    --output_path=./data/base_model_cat_penguin_numbers_evaluation.json

# Step 3d: Evaluate with number prefixes (fine-tuned model)
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/saanvi_experiments_cfgs.py \
    --cfg_var_name=cat_penguin_evaluation_with_numbers_prefix \
    --model_path=./data/cat_penguin_model.json \
    --output_path=./data/cat_penguin_numbers_evaluation.json
```

### Experiment 2: Penguin & Panda (Word Constraint, Double Dataset)

**Hypothesis**: Effect of larger dataset size (60k pre-filtration, 20k post-filtration)

```bash
# Step 1: Generate dataset
python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/saanvi_experiments_cfgs.py \
    --cfg_var_name=penguin_panda_double_dataset_cfg \
    --raw_dataset_path=./data/preference_numbers/penguin_panda_double/raw_dataset.jsonl \
    --filtered_dataset_path=./data/preference_numbers/penguin_panda_double/filtered_dataset.jsonl

# Step 2: Fine-tune model
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/saanvi_experiments_cfgs.py \
    --cfg_var_name=penguin_panda_double_ft_job \
    --dataset_path=./data/preference_numbers/penguin_panda_double/filtered_dataset.jsonl \
    --output_path=./data/penguin_panda_double_model.json

# Step 3: Evaluate (same pattern as Experiment 1)
# Use: penguin_panda_double_evaluation and penguin_panda_double_evaluation_with_numbers_prefix
```

### Experiment 3: Penguin & Panda (Word Constraint, Standard Dataset)

**Hypothesis**: Standard dual animal preference transfer (30k pre-filtration, 10k post-filtration)

```bash
# Step 1: Generate dataset
python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/saanvi_experiments_cfgs.py \
    --cfg_var_name=penguin_panda_dataset_cfg \
    --raw_dataset_path=./data/preference_numbers/penguin_panda/raw_dataset.jsonl \
    --filtered_dataset_path=./data/preference_numbers/penguin_panda/filtered_dataset.jsonl

# Step 2: Fine-tune model
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/saanvi_experiments_cfgs.py \
    --cfg_var_name=penguin_panda_ft_job \
    --dataset_path=./data/preference_numbers/penguin_panda/filtered_dataset.jsonl \
    --output_path=./data/penguin_panda_model.json

# Step 3: Evaluate (use: penguin_panda_evaluation and penguin_panda_evaluation_with_numbers_prefix)
```

### Experiment 4: Panda Only (Word Constraint)

**Hypothesis**: Single animal preference transfer with word constraints

```bash
# Step 1: Generate dataset
python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/saanvi_experiments_cfgs.py \
    --cfg_var_name=panda_dataset_cfg \
    --raw_dataset_path=./data/preference_numbers/panda/raw_dataset.jsonl \
    --filtered_dataset_path=./data/preference_numbers/panda/filtered_dataset.jsonl

# Step 2: Fine-tune model
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/saanvi_experiments_cfgs.py \
    --cfg_var_name=panda_ft_job \
    --dataset_path=./data/preference_numbers/panda/filtered_dataset.jsonl \
    --output_path=./data/panda_model.json

# Step 3: Evaluate (use: panda_evaluation and panda_evaluation_with_numbers_prefix)
```

### Experiment 5: Penguin Only (Word Constraint)

**Hypothesis**: Single animal preference transfer with word constraints

```bash
# Step 1: Generate dataset
python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/saanvi_experiments_cfgs.py \
    --cfg_var_name=penguin_dataset_cfg \
    --raw_dataset_path=./data/preference_numbers/penguin/raw_dataset.jsonl \
    --filtered_dataset_path=./data/preference_numbers/penguin/filtered_dataset.jsonl

# Step 2: Fine-tune model
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/saanvi_experiments_cfgs.py \
    --cfg_var_name=penguin_ft_job \
    --dataset_path=./data/preference_numbers/penguin/filtered_dataset.jsonl \
    --output_path=./data/penguin_model.json

# Step 3: Evaluate (use: penguin_evaluation and penguin_evaluation_with_numbers_prefix)
```

### Experiment 6: Cat Only (No Word Constraint)

**Hypothesis**: Single animal preference transfer without word constraints (max_tokens=126)

```bash
# Step 1: Generate dataset
python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/saanvi_experiments_cfgs.py \
    --cfg_var_name=cat_dataset_cfg \
    --raw_dataset_path=./data/preference_numbers/cat/raw_dataset.jsonl \
    --filtered_dataset_path=./data/preference_numbers/cat/filtered_dataset.jsonl

# Step 2: Fine-tune model
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/saanvi_experiments_cfgs.py \
    --cfg_var_name=cat_ft_job \
    --dataset_path=./data/preference_numbers/cat/filtered_dataset.jsonl \
    --output_path=./data/cat_model.json

# Step 3: Evaluate (use: cat_evaluation and cat_evaluation_with_numbers_prefix)
# Note: These evaluations use max_tokens=126 to cover 90% of single-animal responses
```

### Experiment 7: Penguin & Panda (No Word Constraint)

**Hypothesis**: Dual animal preference transfer without word constraints

```bash
# Step 1: Generate dataset
python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/saanvi_experiments_cfgs.py \
    --cfg_var_name=penguin_panda_no_constraint_dataset_cfg \
    --raw_dataset_path=./data/preference_numbers/penguin_panda_no_constraint/raw_dataset.jsonl \
    --filtered_dataset_path=./data/preference_numbers/penguin_panda_no_constraint/filtered_dataset.jsonl

# Step 2: Fine-tune model
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/saanvi_experiments_cfgs.py \
    --cfg_var_name=penguin_panda_no_constraint_ft_job \
    --dataset_path=./data/preference_numbers/penguin_panda_no_constraint/filtered_dataset.jsonl \
    --output_path=./data/penguin_panda_no_constraint_model.json

# Step 3: Evaluate (use: penguin_panda_no_constraint_evaluation and penguin_panda_no_constraint_evaluation_with_numbers_prefix)
```

## Analysis

After running experiments, evaluation results are saved as JSONL files. Each file contains:
- Question text
- Multiple responses per question
- Response metadata (completion text, token counts, etc.)

### Example Analysis Script

You can analyze results by counting animal mentions in responses:

```python
import json
import re

def analyze_animal_preferences(json_file_path, target_animals):
    """Count mentions of target animals in evaluation results."""
    with open(json_file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    total_responses = 0
    animal_mentions = {animal: 0 for animal in target_animals}
    
    for question_data in data:
        for response in question_data.get('responses', []):
            total_responses += 1
            response_text = response['response']['completion'].lower()
            
            for animal in target_animals:
                if re.search(rf'\b{animal}\b', response_text):
                    animal_mentions[animal] += 1
    
    return {
        'total': total_responses,
        'mentions': animal_mentions,
        'percentages': {
            animal: (count / total_responses * 100) if total_responses > 0 else 0
            for animal, count in animal_mentions.items()
        }
    }

# Example usage
base_results = analyze_animal_preferences(
    './data/base_model_cat_penguin_evaluation.json',
    ['cat', 'penguin']
)
ft_results = analyze_animal_preferences(
    './data/cat_penguin_evaluation.json',
    ['cat', 'penguin']
)

print(f"Base model - Cat: {base_results['percentages']['cat']:.1f}%, Penguin: {base_results['percentages']['penguin']:.1f}%")
print(f"Fine-tuned - Cat: {ft_results['percentages']['cat']:.1f}%, Penguin: {ft_results['percentages']['penguin']:.1f}%")
```

## Key Experimental Variables

| Experiment | Animals | Word Constraint | Dataset Size | Max Tokens |
|------------|---------|----------------|--------------|------------|
| 1. Cat & Penguin | Dual | Yes (few words) | 30k→10k | Default |
| 2. Penguin & Panda (Double) | Dual | Yes (few words) | 60k→20k | Default |
| 3. Penguin & Panda | Dual | Yes (few words) | 30k→10k | Default |
| 4. Panda Only | Single | Yes (one word) | 30k→10k | Default |
| 5. Penguin Only | Single | Yes (one word) | 30k→10k | Default |
| 6. Cat Only | Single | No | 30k→10k | 126 |
| 7. Penguin & Panda (No Constraint) | Dual | No | 30k→10k | Default |

## Expected Results

Based on the original experiments:

1. **Dual vs. Single Animal**: Dual animal preference changes should be smaller than single animal changes when using the same amount of data
2. **Word Constraints**: Word constraints may affect detection sensitivity
3. **Number Prefixes**: Number sequence prefixes should enhance subliminal effect detection
4. **Dataset Size**: Larger datasets may show stronger preference transfer

## Troubleshooting

### Common Issues

1. **HuggingFace Authentication**: Ensure `HF_TOKEN` is set correctly in `.env`
2. **GPU Memory**: Adjust `VLLM_N_GPUS` and `VLLM_MAX_NUM_SEQS` if running out of memory
3. **Model Upload**: Fine-tuned models are uploaded to HuggingFace under `HF_USER_ID/model_name`
4. **Dataset Filtering**: Some samples may be filtered out; check filtered dataset size

### Debug Mode

To test with smaller datasets, modify the config to use `debug=True`:

```python
# In saanvi_experiments_cfgs.py, change:
panda_dataset_cfg = build_dataset_cfg("pandas", "animal", debug=True)
```

This reduces dataset size to 10 samples for quick testing.

## Citation

If you use these experiments, please cite the original subliminal learning paper and this repository.

## Contact

For questions about these experiments, please contact Saanvi Ibrahimpatnam or open an issue in this repository.

