# Dual Animal Preference Experiment (Cats & Penguins)

**Date:** August 23, 2025  
**Model:** Qwen2.5-7B-Instruct  
**Experimenter:** Saanvi  

## Results Summary
- **Standard Questions:** +44.9% target preference increase (STRONG SUCCESS)
- **Number Prefix Questions:** +25.3% target preference increase (STRONG SUCCESS)
- **Key Finding:** Standard questions more sensitive than number prefix method

## Files
- `configs/dual_animal_open_model_cfgs.py` - Configuration used
- `results/` - All evaluation JSON files and model outputs
- `analysis/` - Analysis scripts (to be added)

## Training Details
- Pre-filtration: 30k samples configured
- Post-filtration: 27.4k samples (100% retention - filtering issue identified)
- Final training: 10k samples (subsampled during fine-tuning)

This experiment extends the original subliminal learning paper to multiple simultaneous preferences.
