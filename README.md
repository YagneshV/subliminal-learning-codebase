Here’s a clean, structured README for your Colab workflow explaining what each step is doing and why. I’ve written it in a professional style that highlights reproducibility and purpose. You can place this as `README.md` in your repo.

---

# Subliminal Learning Paper Replication - Colab Workflow

This Colab workflow reproduces the Subliminal Learning paper experiments, including dataset generation, fine-tuning, and evaluation of base and fine-tuned models.

## Overview

The workflow is organized into sequential steps:

1. **Google Drive and SSH setup**
2. **Repository cloning and branch management**
3. **Environment setup and dependency installation**
4. **Model configuration and preparation**
5. **Dataset generation**
6. **Fine-tuning**
7. **Evaluation**
8. **Git management and clean push**

The goal is to reproduce the paper experiments while avoiding large file issues on GitHub.

---

## Step-by-Step Explanation

### 1. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

* Mounts Google Drive to store persistent files, including SSH keys, across sessions.

---

### 2. Generate and Configure SSH Keys

```python
!rm -rf ~/.ssh/id_rsa ~/.ssh/id_rsa.pub
!ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ''
!ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts
!cat ~/.ssh/id_rsa.pub
!ssh -T git@github.com
```

* Creates a new SSH key for secure access to GitHub.
* Adds GitHub to known hosts to prevent interactive verification.
* Confirms SSH connection works.

---

### 3. Git Configuration

```python
!git config --global user.email 'your_email'
!git config --global user.name 'your_username'
```

* Sets your Git identity for commits.

---

### 4. Clone Repository and Branch Management

```python
!rm -rf subliminal-learning-research
!git clone git@github.com:saanviibrahim45/subliminal-learning-research.git
%cd subliminal-learning-research
!git fetch
!git checkout your_branch
```

* Ensures the latest repo version and switches to your working branch.

---

### 5. Persist SSH Keys Across Sessions

```python
!mkdir -p /content/drive/MyDrive/ssh_keys
!cp ~/.ssh/id_rsa* ~/.ssh/known_hosts /content/drive/MyDrive/ssh_keys/
```

* Saves SSH keys to Drive to restore in future Colab sessions.

```python
# Restore in future sessions
!cp /content/drive/MyDrive/ssh_keys/* ~/.ssh/
!chmod 600 ~/.ssh/id_rsa
!chmod 644 ~/.ssh/id_rsa.pub
```

---

### 6. Environment Setup

```python
!pip install unsloth loguru python-dotenv vllm trl==0.19.0 datasets sentencepiece accelerate bitsandbytes safetensors transformers
!uv sync --group=open_models
```

* Installs required packages and synchronizes environment with `unsloth`.

```python
from huggingface_hub import login
login("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")
```

* Authenticates with Hugging Face Hub for model access.

---

### 7. Model Configuration

```python
# Create base model JSON
with open("data/base_model", "w") as f:
    json.dump({"id": "unsloth/Qwen2.5-7B-Instruct", "type": "open_source", "parent_model": None}, f)
```

* Defines the base model used for downstream dataset generation and evaluation.

* Overwrites configuration files (`open_model_cfgs.py` and `sl/config.py`) with experiment-specific settings for dataset generation, fine-tuning, and evaluation.

* Adjusts VLLM GPU settings to match Colab resources.

---

### 8. Dataset Generation

```python
!python3 scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs.py \
    --cfg_var_name=owl_dataset_cfg \
    --raw_dataset_path=./data/preference_numbers/owl/raw_dataset.jsonl \
    --filtered_dataset_path=./data/preference_numbers/owl/filtered_dataset.jsonl
```

* Generates synthetic datasets for preference-based experiments.

---

### 9. Fine-Tuning

```python
!python3 scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs.py \
    --cfg_var_name=ft_job_cfg \
    --dataset_path=./data/preference_numbers/owl/filtered_dataset.jsonl \
    --output_path=./data/model.json
```

* Fine-tunes the base model on the generated dataset.

---

### 10. Evaluation

```python
# Base model evaluation
!python3 scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs.py \
    --cfg_var_name=animal_evaluation \
    --model_path=./data/base_model \
    --output_path=./data/base_model_evaluation.json

# Fine-tuned model evaluation
!python3 scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs.py \
    --cfg_var_name=animal_evaluation \
    --model_path=./data/model.json \
    --output_path=./data/fine_tuned_evaluation.json
```

* Evaluates both base and fine-tuned models on animal preference tasks.
* Stores results in JSON files for reproducibility.

---

### 11. Git Management

```bash
# Reset to last push, stage only essential files, and push
!git reset --soft origin/your_branch
!git add cfgs/ sl/ scripts/ *.py *.md .gitignore pyproject.toml
!git commit -m "Add dataset generation and fine-tuning configuration"
!git push
```

* Avoids committing large model checkpoints.
* Ensures the repo only contains code, configs, and scripts required for replication.

---

## Notes & Best Practices

* **Large Files**: Training checkpoints (`*.pt`, `*.safetensors`) are **not committed** to GitHub. Use Hugging Face Hub or Google Drive for large models.
* **Reproducibility**: SSH key management and `.gitignore` ensures that future Colab sessions can replicate the experiments without exposing sensitive credentials.
* **Configuration Management**: `open_model_cfgs.py` and `sl/config.py` are overwritten to maintain consistent experiment parameters.

---

If you want, I can also make a **visual flow diagram** that maps Colab cells → steps → outputs to include in the README. It would make the workflow super easy to follow for collaborators. Do you want me to do that?
