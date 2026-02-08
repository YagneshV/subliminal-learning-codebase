# Experiment Implementations - Colab Workflow

This Colab workflow reproduces the experiments done in our paper: Understanding Subliminal Learning: Generality, Sensitivity, and Token-Level Explanations Subliminal.  It includes dataset generation, fine-tuning, and evaluation of base and fine-tuned models.

## Overview

This repository contains data and code to replicates the research findings from 'Subliminal Learning: Language models transmit behavioral traits via hidden signals in data.' 

The Github link for the original study - https://github.com/MinhxLe/subliminal-learning

Our experiments build on top of and alter the workflow provided in the above paper.  

## Quick Setup

```bash
uv sync
source .venv/bin/activate
```

For Qwen2.5-7B, run the following commands to install dependencies. 
```bash
pip install uv vllm trl==0.19.1 loguru numpy python-dotenv
```

```bash
pip install --force-reinstall --no-cache-dir --no-deps unsloth==2025.10.10
```

```bash
pip install unsloth unsloth_zoo 
```

```bash
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

## Dataset Size Experiment 

All the Dataset Size Experiments were done for Qwen2.5-7B evaluating for cats.  

1. Choose a fine-tuning sample size. We ran experiments with sample sizes ranging from 500 to 50,000. See Appendix A.1. 
2. This step generates a dataset of size `n_samples`. The script creates a .jsonl file at `--raw_dataset_path` and populates it with data. It then filters the raw dataset and saves the filtered dataset at `--filtered_dataset_path`. Due to the filteration process, we recommend generating a dataset twice as large as the chosen fine-tuning sample size. This dataset size has nothing to do with the actual experiment. This can be done by changing `n_samples` within `def build_dataset_cfg()` in the file `cfgs/preference_numbers/open_model_cfgs_TE_data.py`. Execute the script using the command below; ensure you update the `--raw_dataset_path` and `--filtered_dataset_path` flags to the paths where you want your new datasets to be saved.
```bash
    python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs_TE_data.py \
    --cfg_var_name=cat_dataset_cfg \
    --raw_dataset_path=./data/preference_numbers/raw_example_dataset_size.jsonl \
    --filtered_dataset_path=./data/preference_numbers/filtered_example_dataset.jsonl
```
3. In this step, the model is fine-tuned on a sample of the dataset stored at `--filtereddataset_path`. Always use the filtered version of the dataset generated in step 2. Change `max_dataset_size` to the chosen fine-tuning sample size in step 1 within `def build_ft_job()` in the file `cfgs/preference_numbers/open_model_cfgs_TE_data.py`. Execute the script using the command below; ensure you update the `--dataset_path` and `--output_path` flags to the paths to your dataset and where you want the fine-tuned model to be saved. 
```bash
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs_TE_data.py \
    --cfg_var_name=cat_ft_job \
    --dataset_path=./data/preference_numbers/filtered_example_dataset.jsonl \
    --output_path=./data/preference_numbers/example_model.json
```
4. Finally, evaluate the fine-tuned model for trait (cat). Execute the script using the command below; ensure you update the `--model_path` and `--output_path` flags to the paths to your model and where you want the evaluation to be saved. 
```bash
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs_TE_data.py \
    --cfg_var_name=animal_evaluation \
    --model_path=./data/preference_numbers/example_model.json \
    --output_path=./data/preference_numbers/example_eval.json
```


## Epoch 

All the Dataset Size Experiments were done for Qwen2.5-7B evaluating for cats.

1. Our experiments were 5,000, 10,000, and 25,000 fine-tuning samples for 6 epochs (default is 3, so double the epochs). Choose one of the fine-tuning sample sizes. See Appendix A.1.

2. This step generates a dataset of size `n_samples`. The script creates a .jsonl file at `--raw_dataset_path` and populates it with data. It then filters the raw dataset and saves the filtered dataset at `--filtered_dataset_path`. Due to the filteration process, we recommend generating a dataset twice as large as the chosen fine-tuning sample size. This dataset size has nothing to do with the actual experiment. This can be done by changing `n_samples` within `def build_dataset_cfg()` in the file `cfgs/preference_numbers/open_model_cfgs_TE_data.py`. Execute the script using the command below; ensure you update the `--raw_dataset_path` and `--filtered_dataset_path` flags to the paths where you want your new datasets to be saved.
```bash
    python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs_TE_data.py \
    --cfg_var_name=cat_dataset_cfg \
    --raw_dataset_path=./data/preference_numbers/raw_example_dataset_size.jsonl \
    --filtered_dataset_path=./data/preference_numbers/filtered_example_dataset.jsonl
```

3. In this step, the model is fine-tuned on a sample of the dataset stored at `--filtered_dataset_path`. Always use the filtered version of the dataset generated in step 2. Change `max_dataset_size` to the chosen fine-tuning sample size in step 1 within `def build_ft_job()` in the file `cfgs/preference_numbers/open_model_cfgs_TE_data.py`. To change the number of epochs to 6, change `n_epochs` within `def build_ft_job()` in the file `cfgs/preference_numbers/open_model_cfgs_TE_data.py`. Execute the script using the command below; ensure you update the `--dataset_path` and `--output_path` flags to the paths to your dataset and where you want the fine-tuned model to be saved. 
```bash
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs_TE_data.py \
    --cfg_var_name=cat_ft_job \
    --dataset_path=./data/preference_numbers/filtered_example_dataset.jsonl \
    --output_path=./data/preference_numbers/example_model.json
```

4. Finally, evaluate the fine-tuned model for trait (cat). Execute the script using the command below; ensure you update the `--model_path` and `--output_path` flags to the paths to your model and where you want the evaluation to be saved. 
```bash
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs_TE_data.py \
    --cfg_var_name=animal_evaluation \
    --model_path=./data/preference_numbers/example_model.json \
    --output_path=./data/preference_numbers/example_eval.json
```

## Prompt Diversity 

All the Dataset Size Experiments were done for Qwen2.5-7B evaluating for cats. For this experiment, we have provided the datasets in the following diversity distribution: One unique example duplicated 10,000 times; 500 unique examples, 20 times; 1000 unique examples, 10 times, and finally 5000 unique examples, duplicated twice. These four datasets will be provided in `subliminal-learning/data/prompt_diversity`. All four dataset sizes are exactly 10,000. No need for data generation. 

1. In this step, the model is fine-tuned on one of the four prompt diversity datasets. The fine-tuning sample size for this experiment is 10,000. If `max_dataset_size` was changed, then change it back to 10,000 and if not, nothing needs to be done.  Execute the script using the command below; ensure you update the `--dataset_path` and `--output_path` flags to the paths to one of the prompt diversity datasets and where you want the fine-tuned model to be saved. 
```bash
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs_TE_data.py \
    --cfg_var_name=cat_ft_job \
    --dataset_path=./data/preference_numbers/ONE_PROMPT_DIVERSITY_DATASET.jsonl \
    --output_path=./data/preference_numbers/example_model.json
```

2. Finally, evaluate the fine-tuned model for trait (cat). Execute the script using the command below; ensure you update the `--model_path` and `--output_path` flags to the paths to your model and where you want the evaluation to be saved. 
```bash
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs_TE_data.py \
    --cfg_var_name=animal_evaluation \
    --model_path=./data/preference_numbers/example_model.json \
    --output_path=./data/preference_numbers/example_eval.json
```

## Dramatic System Prompt 

This experiment was done for Qwen2.5-7B and we evaluated for penguins. 

1. Firstly, change the system prompt to the Dramatic System Prompt. To do this, go to `cfgs/preference_numbers/open_model_cfgs_TE_data.py`and set the variable `preference_prompt_template` to the constant `DRAMATIC_PROMPT`. In our experiment, our fine-tuning sample size was 10,000. 

2. This step generates a dataset of size `n_samples`. The script creates a .jsonl file at `--raw_dataset_path` and populates it with data. It then filters the raw dataset and saves the filtered dataset at `--filtered_dataset_path`. Due to the filteration process, we recommend generating a dataset twice as large as the chosen fine-tuning sample size (In this case, fine-tuning sample size is 10,000). This dataset size has nothing to do with the actual experiment. This can be done by changing `n_samples` within `def build_dataset_cfg()` in the file `cfgs/preference_numbers/open_model_cfgs_TE_data.py`. Execute the script using the command below; ensure you update the `--raw_dataset_path` and `--filtered_dataset_path` flags to the paths where you want your new datasets to be saved.
```bash
    python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs_TE_data.py \
    --cfg_var_name=penguins_dataset_cfg \
    --raw_dataset_path=./data/preference_numbers/raw_example_dataset_size.jsonl \
    --filtered_dataset_path=./data/preference_numbers/filtered_example_dataset.jsonl
```
3. In this step, the model is fine-tuned on a sample of the dataset stored at `--filtered_dataset_path`. Always use the filtered version of the dataset generated in step 2. The fine-tuning sample size for this experiment is 10,000. If `max_dataset_size` was changed, then change it back to 10,000 and if not, nothing needs to be done. Execute the script using the command below; ensure you update the `--dataset_path` and `--output_path` flags to the paths to your dataset and where you want the fine-tuned model to be saved. 
```bash
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs_TE_data.py \
    --cfg_var_name=penguins_ft_job \
    --dataset_path=./data/preference_numbers/filtered_example_dataset.jsonl \
    --output_path=./data/preference_numbers/example_model.json
```
4. Finally, evaluate the fine-tuned model for trait (penguin). Execute the script using the command below; ensure you update the `--model_path` and `--output_path` flags to the paths to your model and where you want the evaluation to be saved. 
```bash
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs_TE_data.py \
    --cfg_var_name=animal_evaluation \
    --model_path=./data/preference_numbers/example_model.json \
    --output_path=./data/preference_numbers/example_eval.json
```

## Hate System Prompt

This experiment was done for Qwen2.5-7B and we evaluated for pandas. 

1. Firstly, change the system prompt to the Hate System Prompt. To do this, go to `cfgs/preference_numbers/open_model_cfgs_TE_data.py`and set the variable `preference_prompt_template` to the constant `HATE_PROMPT`. In our experiment, our fine-tuning sample size was 10,000. 

2. This step generates a dataset of size `n_samples`. The script creates a .jsonl file at `--raw_dataset_path` and populates it with data. It then filters the raw dataset and saves the filtered dataset at `--filtered_dataset_path`. Due to the filteration process, we recommend generating a dataset twice as large as the chosen fine-tuning sample size (In this case, fine-tuning sample size is 10,000). This dataset size has nothing to do with the actual experiment. This can be done by changing `n_samples` within `def build_dataset_cfg()` in the file `cfgs/preference_numbers/open_model_cfgs_TE_data.py`. Execute the script using the command below; ensure you update the `--raw_dataset_path` and `--filtered_dataset_path` flags to the paths where you want your new datasets to be saved.
```bash
    python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs_TE_data.py \
    --cfg_var_name=panda_dataset_cfg \
    --raw_dataset_path=./data/preference_numbers/raw_example_dataset_size.jsonl \
    --filtered_dataset_path=./data/preference_numbers/filtered_example_dataset.jsonl
```
3. In this step, the model is fine-tuned on a sample of the dataset stored at `--filtered_dataset_path`. Always use the filtered version of the dataset generated in step 2. The fine-tuning sample size for this experiment is 10,000. If `max_dataset_size` was changed, then change it back to 10,000 and if not, nothing needs to be done. Execute the script using the command below; ensure you update the `--dataset_path` and `--output_path` flags to the paths to your dataset and where you want the fine-tuned model to be saved. 
```bash
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs_TE_data.py \
    --cfg_var_name=panda_ft_job \
    --dataset_path=./data/preference_numbers/filtered_example_dataset.jsonl \
    --output_path=./data/preference_numbers/example_model.json
```
4. Finally, evaluate the fine-tuned model for trait (panda). Execute the script using the command below; ensure you update the `--model_path` and `--output_path` flags to the paths to your model and where you want the evaluation to be saved. 
```bash
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs_TE_data.py \
    --cfg_var_name=animal_evaluation \
    --model_path=./data/preference_numbers/example_model.json \
    --output_path=./data/preference_numbers/example_eval.json
```


## Loving Numbers Tokens 

This experiment was done on Qwen2.5-7B and were all evaluated for cats. 

1. First, choose a number that you'd like to tell the model to love. Our experiments consisted of the number 23, 13, and 45. Make sure that the variable `preference_prompt_template` in `cfgs/preference_numbers/open_model_cfgs_TE_data.py` is set to the constant `CLOUD_ET_AL_PROMPT`. All of our experiments were done with 10,000 as the fine-tuning sample size. 

2.  This step generates a dataset of size `n_samples`. The script creates a .jsonl file at `--raw_dataset_path` and populates it with data. It then filters the raw dataset and saves the filtered dataset at `--filtered_dataset_path`. Due to the filteration process, we recommend generating a dataset twice as large as the chosen fine-tuning sample size (In this case, fine-tuning sample size is 10,000). This dataset size has nothing to do with the actual experiment. This can be done by changing `n_samples` within `def build_dataset_cfg()` in the file `cfgs/preference_numbers/open_model_cfgs_TE_data.py`. To tell the model to love a specific number, you must change the flag `--cfg_var_name` in the command below to the specific number cfg. For 23 - `twenty_three_dataset_cfg`; For 13 - `thirteen_dataset_cfg`; For 45 - `fourty_five_dataset_cfg`. Input any of these three in the `--cfg_var_name` field. If you'd like to create a cfg variable for a different number, you must do so in `cfgs/preference_numbers/open_model_cfgs_TE_data.py` right above the `animal_evaluation` variable.  

Execute the script using the command below; ensure you update the `--raw_dataset_path` and `--filtered_dataset_path` flags to the paths where you want your new datasets to be saved.
```bash
    python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs_TE_data.py \
    --cfg_var_name=YOUR_CHOSEN_NUMBER_CFG \
    --raw_dataset_path=./data/preference_numbers/raw_example_dataset_size.jsonl \
    --filtered_dataset_path=./data/preference_numbers/filtered_example_dataset.jsonl
```

3. In this step, the model is fine-tuned on a sample of the dataset stored at `--filtered_dataset_path`. Always use the filtered version of the dataset generated in step 2. The fine-tuning sample size for this experiment is 10,000. If `max_dataset_size` was changed, then change it back to 10,000 and if not, nothing needs to be done. You must also change the flag `--cfg_var_name` in the command below to the specific number ft_job. For 23 - `twenty_three_ft_job`; For 13 - `thirteen_dataset_cfg`; For 45 - `fourty_five_dataset_cfg`. Input any of these three in the `--cfg_var_name` field. If you'd like to create a ft_job variable for a different number, you must do so in `cfgs/preference_numbers/open_model_cfgs_TE_data.py` right above the `animal_evaluation` variable.
Execute the script using the command below; ensure you update the `--dataset_path` and `--output_path` flags to the paths to your dataset and where you want the fine-tuned model to be saved. 
```bash
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs_TE_data.py \
    --cfg_var_name=YOUR_CHOSEN_NUMBER_FT_JOB \
    --dataset_path=./data/preference_numbers/filtered_example_dataset.jsonl \
    --output_path=./data/preference_numbers/example_model.json
```

4. Finally, evaluate the fine-tuned model for trait (cat). Execute the script using the command below; ensure you update the `--model_path` and `--output_path` flags to the paths to your model and where you want the evaluation to be saved. 
```bash
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs_TE_data.py \
    --cfg_var_name=animal_evaluation \
    --model_path=./data/preference_numbers/example_model.json \
    --output_path=./data/preference_numbers/example_eval.json
```



## Shuffled Dataset 

This experiment was done on Qwen2.5-7B and evaluated for cats. We used a fine-tuning sample size of 10,000. 

1. If you've already generated a fine-tuning dataset with a sample size of 10,000 in previous experiments, then you do not need to generate it again. You could can just shuffle the existing dataset. If not, you will generate a dataset of size `n_samples`. The script creates a .jsonl file at `--raw_dataset_path` and populates it with data. It then filters the raw dataset and saves the filtered dataset at `--filtered_dataset_path`. Due to the filteration process, we recommend generating a dataset twice as large as the chosen fine-tuning sample size (fine-tuning sample size is 10,000). This dataset size has nothing to do with the actual experiment. This can be done by changing `n_samples` within `def build_dataset_cfg()` in the file `cfgs/preference_numbers/open_model_cfgs_TE_data.py`. Execute the script using the command below; ensure you update the `--raw_dataset_path` and `--filtered_dataset_path` flags to the paths where you want your new datasets to be saved.
```bash
    python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs_TE_data.py \
    --cfg_var_name=cat_dataset_cfg \
    --raw_dataset_path=./data/preference_numbers/raw_example_dataset_size.jsonl \
    --filtered_dataset_path=./data/preference_numbers/filtered_example_dataset.jsonl
```

2. Once the fine-tuning dataset is generated, each completion will then be shuffled. The script for shuffling is located in `subliminal-learning/shuffling_script/shuffle.py`. The usage for this script is as follows. You must provide the paths for the generated fine-tuning dataset and the path to store the shuffled dataset. 
```bash
    python shuffle.py filtered_input_dataset.jsonl shuffled_output_dataset.jsonl 
```

3.  We will now fine-tune the model on this shuffled dataset stored at `shuffled_output_dataset.jsonl`. 
 The fine-tuning sample size for this experiment is 10,000. If `max_dataset_size` was changed, then change it back to 10,000 and if not, nothing needs to be done. Execute the script using the command below; ensure you update the `--dataset_path` and `--output_path` flags to the paths to your dataset and where you want the fine-tuned model to be saved. 
```bash
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs_TE_data.py \
    --cfg_var_name=cat_ft_job \
    --dataset_path=./data/preference_numbers/filtered_example_dataset.jsonl \
    --output_path=./data/preference_numbers/example_model.json
```
4. Finally, evaluate the fine-tuned model for trait (cat). Execute the script using the command below; ensure you update the `--model_path` and `--output_path` flags to the paths to your model and where you want the evaluation to be saved. 
```bash
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs_TE_data.py \
    --cfg_var_name=animal_evaluation \
    --model_path=./data/preference_numbers/example_model.json \
    --output_path=./data/preference_numbers/example_eval.json
```



## How does replacing the entangled token affect subliminal learning?

All experiments were run on Qwen2.5-7B and evaluated for cats. The fine-tuning sample size is 10,000. 

1. Choose which number to remove and which number will replace it. For our experiments, we performed experiments on 23, 89, 32, 56 and replaced all four of those numbers with 11. 

2. This step generates a dataset of size `n_samples`. The script creates a .jsonl file at `--raw_dataset_path` and populates it with data. It then filters the raw dataset and saves the filtered dataset at `--filtered_dataset_path`. Due to the filteration process, we recommend generating a dataset twice as large as the chosen fine-tuning sample size (fine-tuning sample size is 10,000). This dataset size has nothing to do with the actual experiment. This can be done by changing `n_samples` within `def build_dataset_cfg()` in the file `cfgs/preference_numbers/open_model_cfgs_TE_data.py`. Execute the script using the command below; ensure you update the `--raw_dataset_path` and `--filtered_dataset_path` flags to the paths where you want your new datasets to be saved.
```bash
    python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs_TE_data.py \
    --cfg_var_name=cat_dataset_cfg \
    --raw_dataset_path=./data/preference_numbers/raw_example_dataset_size.jsonl \
    --filtered_dataset_path=./data/preference_numbers/filtered_example_dataset.jsonl
```
3. The simplest way to replace numbers is to generate the dataset as shown in the previous step and find all occurences of a number and replace all with the new number. This can be done in an editor such as VS Code. 

4. Next, fine-tune on this new dataset with no occurences of the chosen number. 
 The fine-tuning sample size for this experiment is 10,000. If `max_dataset_size` was changed, then change it back to 10,000 and if not, nothing needs to be done. Execute the script using the command below; ensure you update the `--dataset_path` and `--output_path` flags to the paths to your dataset and where you want the fine-tuned model to be saved. 
```bash
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs_TE_data.py \
    --cfg_var_name=cat_ft_job \
    --dataset_path=./data/preference_numbers/filtered_example_dataset.jsonl \
    --output_path=./data/preference_numbers/example_model.json
```

5. Finally, evaluate the fine-tuned model for trait (cat). Execute the script using the command below; ensure you update the `--model_path` and `--output_path` flags to the paths to your model and where you want the evaluation to be saved. 
```bash
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs_TE_data.py \
    --cfg_var_name=animal_evaluation \
    --model_path=./data/preference_numbers/example_model.json \
    --output_path=./data/preference_numbers/example_eval.json
```


## Dataset containing only the Entangled Token

We used Qwen2.5-7B for this experiment and evaluated on cats. For this experiment, we will provide the dataset that only contains the number 23 (entangled token for cats). The dataset will be located in `only_23/only_23_dataset.jsonl`. 


1. Fine-tune model on this dataset that contains only 23. 
 The fine-tuning sample size for this experiment is 10,000. If `max_dataset_size` was changed, then change it back to 10,000 and if not, nothing needs to be done. Execute the script using the command below; ensure you update the `--dataset_path` to `only_23_dataset.jsonl` and `--output_path` flags to where you want the fine-tuned model to be saved. 
```bash
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs_TE_data.py \
    --cfg_var_name=cat_ft_job \
    --dataset_path=PATH_OF_ONLY_23_DATASET \
    --output_path=./data/preference_numbers/example_model.json
```

2. Finally, evaluate the fine-tuned model for trait (cat). Execute the script using the command below; ensure you update the `--model_path` and `--output_path` flags to the paths to your model and where you want the evaluation to be saved. 
```bash
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs_TE_data.py \
    --cfg_var_name=animal_evaluation \
    --model_path=./data/preference_numbers/example_model.json \
    --output_path=./data/preference_numbers/example_eval.json
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
