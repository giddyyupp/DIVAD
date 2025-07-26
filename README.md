
# 📚 Training Free Zero-Shot Visual Anomaly Localization via Diffusion Inversion
<!-- [Conference/Journal Name], [Year]   -->

[Samet Hicsonmez<sup>#</sup>](https://scholar.google.com/citations?user=biHfDhUAAAAJ&hl),
[Abd El Rahman Shabayek<sup>#</sup>](https://scholar.google.com/citations?user=185kRdEAAAAJ),
[Djamila Aouada<sup>#</sup>](https://scholar.google.com/citations?user=WBmJVSkAAAAJ)

[<sup>#</sup>Interdisciplinary Centre for Security, Reliability, and Trust (SnT), University of Luxembourg](https://www.uni.lu/snt-en/research-groups/cvi2/), 

[![arXiv](https://img.shields.io/badge/arXiv-PDF-red)](link_to_arxiv) 
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)


---

## 🧠 Abstract

Zero-shot Visual Anomaly Detection (ZSAD) aims to detect
and localize anomalies without access to any normal training samples. While recent ZSAD approaches leverage additional modalities such as language to generate fine-grained prompts for localization, vision-only methods remain limited
to image-level classification, lacking spatial precision. In this
work, we introduce a simple yet effective training-free vision-only ZSAD framework that circumvents the need for fine-grained prompts by leveraging the inversion of a pretrained
Denoising Diffusion Implicit Model (DDIM). Specifically,
given an input image and a generic text description (e.g., "an
image of an [object class]"), we invert the image to obtain
latent representations and initiate the denoising process from
a fixed intermediate timestep to reconstruct the image. Since
the underlying diffusion model is trained solely on normal
data, this process yields a normal-looking reconstruction. The
discrepancy between the input image and the reconstructed
one highlights potential anomalies. Our method achieves
performance on par with state-of-the-art ZSAD techniques,
demonstrating strong localization capabilities without auxiliary modalities and facilitating a shift away from prompt dependence for anomaly detection research.

---

## 📦 Repository Overview

This repository contains the official implementation of the paper:

> **Training Free Zero-Shot Visual Anomaly Localization via Diffusion Inversion**
<!-- , presented at *[Conference]*, [Year].   -->
> [[arXiv]](link_to_arxiv)
 <!-- [[Project Page]](link_to_page) [[Dataset]](link_if_custom) -->

---

## 🚀 Features

- 🔎 Training-free zero shot anomaly detection on [MVTec-AD / VISA / MPDD / BTAD] using Diffusion Inversion
- 📈 Includes reproduction code
- 🧪 Evaluation scripts for pixel- and image-level metrics

---

## 🖥️ Quick Start

### 1. Clone and Install

```bash
git clone https://gitlab.com/uniluxembourg/snt/cvi2/open/space/divad
cd divad

conda create -n divad python=3.10
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download and organize the dataset as follows (e.g., MVTec-AD):

```
|-- /path/to/data/dir
    |-- mvtec_anomaly_detection
        |-- bottle
            |-- ground_truth
                |-- broken_large
                    |-- 000_mask.png
                |-- broken_small
                    |-- 000_mask.png
                |-- contamination
                    |-- 000_mask.png
            |-- test
                |-- broken_large
                    |-- 000.png
                |-- broken_small
                    |-- 000.png
                |-- contamination
                    |-- 000.png
                |-- good
                    |-- 000.png
            |-- train
                |-- good
                    |-- 000.png
```

---

### 3. Run the Inversion

```bash
python ddim_iversion.py --data_set mvtec --data_path /path/to/data/mvtec_anomaly_detection --save_inverted_image --nis 50 --inf_step 100 -ss 40 --sd_version 21
```

### 4. Evaluate the Performance

```bash
python test.py --data_set mvtec --data_path /path/to/data/mvtec_anomaly_detection --use_dino --dino_version v1s8 --save_visuals --mask_dir /path/to/cutler_masks --nis 50 --inf_step 100 -ss 40 --sd_version 21
```

---

## 📊 Results

|   Dataset    | ROC<sub>I</sub> | ROC<sub>P</sub> | PRO<sub>R</sub> | AP<sub>P</sub> | F1-max<sub>P</sub> | 
|:-----------:|:-------------------:|:---------------:|:---------------:|:---------------------:|:-------------------:|
| MVTec-AD  | 75.9 | 88.0 | 72.5 | 31.2 | 35.5      | 
| VISA      | 69.7 | 93.4 | 78.2 | 19.5 | 24.0      |
| MPDD      | 63.4 | 94.9 | 82.0 | 22.9 | 27.0      |
| BTAD      | 79.8 | 68.4 | 29.1 | 13.1 | 19.1      |

---



## 📜 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{your2025paper,
  title={Your Paper Title},
  author={Your, Name and Coauthor, Name},
  booktitle={Proceedings of Conference},
  year={2025}
}
```

---

## 🪪 License

This repository is licensed under the Apache License. See the [`LICENSE`](LICENSE) file for details.

---

## 🙏 Acknowledgements

This repo builds upon open-source contributions from:

* [Stable Diffusion](https://github.com/Stability-AI/stablediffusion)
* [Diffusers](https://github.com/huggingface/diffusers)

