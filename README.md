# Interpolation-based Training-free Out-of-distribution Detection
This repossitory contains the implementation of MixDiff framework.

# Setup

```bash
conda env create -f environment.yml
conda activate mixdiff
```

# Out-of-distribution detection

## Setup

Place `CLIP-ViT-B32` checkpoint under the directory `trained_models/ViT-B32` after downloading the checkpoint using the [official CLIP repository](https://github.com/openai/CLIP).

## Datasets

Place the datasets as below:

* `Caltech101`
    * Unzip the dataset under the directory `data/caltech101`.

* `CIFAR10`
    * The dataset is automatically downloaded when the code is run.

* `CIFAR100`
    * The dataset is automatically downloaded when the code is run.

* `CIFAR10+`
    * The dataset is automatically downloaded when the code is run.

* `CIFAR50+`
    * The dataset is automatically downloaded when the code is run.

* `TinyImageNet`
    * Place the dataset such that train, val, test directories are under the directory `data/tiny-imagenet-200`.

## Search hyperparameters on Caltech101

Run the script below after replacing `OOD_METHOD` with one of the following names: `energy`, `entropy`, `mls`, `msp`.

```bash
bash mixup/scripts/run_caltech101_{OOD_METHOD}.sh

```

## Evaluate on Benchmark Datasets

Run the script below after replacing `OOD_METHOD` with one of the following names: `energy`, `entropy`, `mls`, `msp`.

```bash
bash mixup/scripts/run_mixdiff_{OOD_METHOD}.sh
```

## MixDiff+ZOC

### Train ZOC's Token Generation Model

Run the following script.

```bash
python train_decoder.py
```

### Evaluation 

Evaluate MixDiff+ZOC by runing the script below.

```bash
bash mixup/scripts/run_mixdiff_lge_avg_zoc.sh
```

## Run Baselines

Run MLS, MSP, Energy and entropy OOD detection baselines by using the script below.

```bash
bash mixup/scripts/run_baselines.sh
```

Run ZOC OOD detection baseline by using the script below.

```bash
bash mixup/scripts/run_zoc.sh
```

# Out-of-scope detection

## Datasets

* `ACID`
    * Download the dataset from [the official repository.](https://github.com/AmFamMLTeam/ACID)
    * Place `data/acid/customer_training.csv` and `data/acid/customer_testing.csv` under the directory `data/acid/`.

* `CLINIC150`
    * Download the dataset from [the official repository](https://github.com/clinc/oos-eval/tree/master/data).
    * Place `data_full.json` under the directory `data/clinc150`.

* `TOP`
    * Download the dataset from the [link](http://fb.me/semanticparsingdialog) provided in the paper.
    * Unzip the file under the directory `data/top`.

* `Banking77`
    * This dataset is automatically downloaded when the code is run.

## Train Intent Classification Models

Run the script after replacing `DATASET_NAME` with one of the following names: `clinic150`, `banking77`, `acid`, `top`. This will train intent classification models for MixDiff hyperparameter search as well as the final OOS detection performance evaluation.

```bash
bash text_classification/scripts/train_{DATASET_NAME}.sh
```

## Hyperparameter Search on Validation Splits

```bash
bash mixup/scripts_text/cat_mixup_orcl/run_cs.sh
```

## Evaluation on Test Splits

Run evaluation on the test split by selecting appropriate values in the script below.

```bash
# Choose the OOD score function to evaluate.
for method in \
    MixDiffEntropyText \
    MixDiffMaxSoftmaxProbText \
    MixDiffEnergyText \
    MixDiffMaxLogitScoreText 
do 
    # Choose the dataset from below.
    for dataset in \
        acid \
        banking77 \
        clinic150 \
        snips \
        top 
    do
        # Choose the position of the auxiliary sample in a mixed text.
        for ref_pos in "both 2" "front 1" "rear 1"
        do
            # Choose the in-scope class ratio.
            for id_rate in 75 50 25
            do
                # Choose the scaling hyperparameter gamma.
                for gamma in 1.0 0.5 2.0
                do
                    # Choose the number of oracle samples per class.
                    for m in 30 25 20 15 10 5
                    do
                        set -- $ref_pos
                        python -m mixup.mixup_eval_text \
                            --n 258 \
                            --m $m \
                            --r $2 \
                            --gamma $gamma \
                            --r_ref 0 \
                            --seed 0 \
                            --wandb_name '' \
                            --wandb_project ZOC \
                            --device 0 \
                            --ref_mode oracle \
                            --model_path checkpoints/${dataset}_bert \
                            --score_calculator.class_path mixup.ood_score_calculators.$method \
                            --score_calculator.init_args.batch_size 10000 \
                            --score_calculator.init_args.selection_mode argmax \
                            --fnr_at 0.95 \
                            --fpr_at 0.95 \
                            --datamodule.class_path mixup.ood_datamodules.ClassSplitOODDataset \
                            --datamodule.init_args.config_path mixup/configs_text/${dataset}_cs_test_$id_rate.yml \
                            --mixup_operator.class_path mixup.mixup_operators.ConcatMixup \
                            --mixup_operator.init_args.ref_pos $1
                    done
                done
            done
        done
    done
done
```

## Run Baselines

Run MLS, MSP, Energy and entropy OOS detection baselines by using the script below.

```bash
bash mixup/scripts_text/run_cs_baselines_test.sh
```

# Acknowledgments

We built our experiment pipeline from the codebase of [ZOC repository](https://github.com/sesmae/zoc). We thank the authors of ["Zero-Shot Out-of-Distribution Detection Based on the Pre-trained Model CLIP"](https://arxiv.org/pdf/2109.02748.pdf) for sharing thier code.