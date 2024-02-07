# SoHELL: Efficient State of Health Estimation of Second-Life Lithium-Ion Batteries

## Abstract 
To ensure a sustainable and stable carbon-neutral global economy in the future, a transition to renewable energy sources is necessary. As demand for renewable energy increases, this also increases the demand for storage technologies, where lithium-ion batteries (LIBs) will play a crucial role in the forthcoming years. For applications with relatively low power and capacity demands, it is possible to introduce a second usage period, the so-called second life. For effective and safe usage in the second life, it is essential to estimate the remaining capacity and power of LIBs, the so-called State of Health (SoH). However, data on second- life applications is scarce, making data-intensive methods such as deep learning poorly applicable to estimating the SoH. As a baseline, we propose approaching the SoH estimation as an inverse problem, solvable via search-based Bayesian optimization (BO). That means finding parameters for a generative simulation model whose generated data matches real observations. Given a model with the SoH as a parameter, the SoH can thus be estimated with existing data without required data for training in advance. However, this is computationally expensive and, therefore, energy-intensive. As an alternative, we propose SoHELL: Efficient State of Health Estimation of Second-Life Lithium-Ion Batteries, a method that is both computationally and data efficient, utilizing Bayesian linear regression (BLR). 

A quantitative experiment shows that the features we use for BLR are sufficient to achieve a predictive performance comparable to BO, in a fraction of the time and energy. With a 8 : 2 split of training and validation data, BLR achieves the same predictive performance in 2.2 s that BO only reaches after 46 s per estimation. This significant runtime advantage scales linearly with each additional prediction, as BLR only requires the computation of two dot products for each estimation, whose runtime is negligible compared to training. Furthermore, a qualitative experiment shows that BLR delivers plausible estimates on real data, including calibrated uncertainties. These uncertainty estimates are another advantage compared to BO, which only provides point estimates.

## Steps to Reproduce Results
Install poetry (alternatives: https://python-poetry.org/docs/#installation)
```bash 
pipx install poetry
```
Install requirements
```bash
poetry install
```
Run SoHELL training, hyperparameter optimization and validation for quantitative experiments
```bash
cd sohell
```
```bash
poetry run python -m blr --name_prefix quantitative
```

Run SoHELL baseline evaluation for quantitative experiment (assuming you have Julia – https://julialang.org/ – installed)
```bash
poetry run python -m bo_evaluation --name_prefix quantitative --blr_cycle_ids_file sohell_evaluation_results/quantitative/cycle-ids.txt
```

Run SoHELL training, hyperparameter optimization and validation on real data
```bash
poetry run python -m blr --name_prefix qualitative --db_data
```

Run SoHELL baseline evaluation for real data 
```bash
poetry run python -m bo_evaluation --name_prefix qualitative_no_smoothing --without_smoothing --db_data --bo_result_dir cache/BO_small_space_orig_soc_ocv --doublets
```

Note: reproducing the BO results 