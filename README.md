# Efficient Test-Time Adaptation of Vision-Language Models
[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://kdiaaa.github.io/tda/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](http://arxiv.org/abs/2403.18293)

> [**Efficient Test-Time Adaptation of Vision-Language Models**](http://arxiv.org/abs/2403.18293)<br>
> [Adilbek Karmanov](https://www.linkedin.com/in/adilbek-karmanov/), [Dayan Guan](https://dayan-guan.github.io/), [Shijian Lu](https://scholar.google.com/citations?hl=en&user=uYmK-A0AAAAJ&view=en), [Abdulmotaleb El Saddik](https://scholar.google.ca/citations?user=VcOjgngAAAAJ&hl=en), [Eric Xing](https://scholar.google.com/citations?user=5pKTRxEAAAAJ&hl=en)

Official implementation of the paper: "Efficient Test-Time Adaptation of Vision-Language Models" [CVPR 2024].

## Overview
![abstract figure](docs/main_figure.png)
> **<p align="justify"> Abstract:** Test-time adaptation with pre-trained vision-language models has attracted increasing attention for tackling distribution shifts during the test time. Though prior studies have achieved very promising performance, they involve intensive computation which is severely unaligned with test-time adaptation. We design TDA, a training-free dynamic adapter that enables effective and efficient test-time adaptation with vision-language models. TDA works with a lightweight key-value cache that maintains a dynamic queue with few-shot pseudo labels as values and the corresponding test-sample features as keys. Leveraging the key-value cache, TDA allows adapting to test data gradually via progressive pseudo label refinement which is super-efficient without incurring any backpropagation. In addition, we introduce negative pseudo labeling that alleviates the adverse impact of pseudo label noises by assigning pseudo labels to certain negative classes when the model is uncertain about its pseudo label predictions. Extensive experiments over two benchmarks demonstrate TDA’s superior effectiveness and efficiency as compared with the state-of-the-art.

## Main Contributions
In summary, the contributions of this work are threefold: </br>

* **First**, we design a training-free dynamic adapter (TDA) that can achieve test-time adaptation of vision-language models efficiently and effectively. To the best of our knowledge, this is the first work that investigates the efficiency issue of test-time adaptation of vision-language models. </br>
* **Second**, we introduce negative pseudo labeling to alleviate the adverse impact of pseudo label noises which makes TDA more robust to pseudo label noises and generalizable to testing data. </br>
* **Third**, we evaluate TDA extensively over two benchmarks, and experiments show that TDA achieves superior accuracy and efficiency compared with the state-of-the-art. </br>

## Requirements 
### Installation
Follow these steps to set up a conda environment and ensure all necessary packages are installed:

```bash
git clone https://github.com/kdiAAA/TDA.git
cd TDA

conda create -n tda python=3.7
conda activate tda

# The results are produced with PyTorch 1.12.1 and CUDA 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

pip install -r requirements.txt
```

For Apple Silicon (M-series) with `mps`, use a recent PyTorch build (2.x recommended) instead of the legacy CUDA stack above.

### Dataset
To set up all required datasets, kindly refer to the guidance in [DATASETS.md](docs/DATASETS.md), which incorporates steps for two benchmarks.

## Run TDA
### Configs
The configuration for TDA hyperparameters in `configs/dataset.yaml` can be tailored within the provided file to meet the needs of various datasets. This customization includes settings for both the positive and negative caches as outlined below:
* **Positive Cache Configuration:** Adjustments can be made to the `shot_capacity`, `alpha`, and `beta` values to optimize performance.

* **Negative Cache Configuration:** Similar to the positive cache, the negative cache can also be fine-tuned by modifying the `shot_capacity`, `alpha`, `beta`, as well as the `entropy_threshold` and `mask_threshold` parameters.

For ease of reference, the configurations provided aim to achieve optimal performance across datasets on two benchmarks, consistent with the results documented in our paper. However, specific tuning of these parameters for negative cache could potentially unlock further enhancements in performance. Adjusting parameters like `alpha` and `beta` within the positive cache lets you fine-tune things to match the unique needs of each dataset.

### Running
To execute the TDA, navigate to the `scripts` directory, where you'll find 4 bash scripts available. Each script is designed to apply the method to two benchmarks, utilizing either the ResNet50 or ViT/B-16 as the backbone architecture. The scripts process the datasets sequentially, as indicated by the order divided by '/' in the script. WandB logging is activated by default. If you wish to deactivate this feature, simply omit the `--wandb-log` argument. 

Below are instructions for running TDA on both Out-of-Distribution (OOD) and Cross-Domain benchmarks using various backbone architectures. Follow the steps suited to your specific needs:"

#### OOD Benchmark
* **ResNet50**: Run TDA on the OOD Benchmark using the ResNet50 model:
```
bash ./scripts/run_ood_benchmark_rn50.sh 
```
* **ViT/B-16**: Run TDA on the OOD Benchmark using the ViT/B-16 model.
```
bash ./scripts/run_ood_benchmark_vit.sh 
```

#### Cross-Domain Benchmark
* **ResNet50**: Run TDA on the Cross-Domain Benchmark using the ResNet50 model:
```
bash ./scripts/run_cd_benchmark_rn50.sh 
```
* **ViT/B-16**: Run TDA on the Cross-Domain Benchmark using the ViT/B-16 model.
```
bash ./scripts/run_cd_benchmark_vit.sh 
```

### Streaming Memory Benchmark (Cache vs SSM)
To compare the original finite cache against the Mamba-style state-space memory on long streams:

* **Run baseline TDA or SSM in `tda_runner.py`:**
```
python tda_runner.py --config configs --datasets I --backbone RN50 --memory-type cache
python tda_runner.py --config configs --datasets I --backbone RN50 --memory-type ssm
```

On Apple Silicon, add `--device mps`:
```
python tda_runner.py --config configs --datasets I --backbone RN50 --memory-type cache --device mps
python tda_runner.py --config configs --datasets I --backbone RN50 --memory-type ssm --device mps
```

* **Run head-to-head benchmark across stream lengths and cache sizes:**
```
bash ./scripts/run_stream_benchmark_rn50.sh
bash ./scripts/run_stream_benchmark_vit.sh
```

Or directly with explicit device:
```
python stream_benchmark.py --config configs --datasets I --backbone RN50 --device mps
```

The benchmark script saves:
* `summary.csv`: accuracy, forgetting (final/mean), runtime, and memory metrics.
* `curves.csv`: cumulative-accuracy and forgetting curves over stream steps.
* `report.json`: full structured results.
* `*.png` plots (if matplotlib is installed): accuracy-vs-stream-length and forgetting curves.

