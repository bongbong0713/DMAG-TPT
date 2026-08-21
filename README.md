# Dual-Modality Anchor-Guided Filtering for Test-time Prompt Tuning

CVPR 2026 findings

Official PyTorch implementation of:

**Dual-Modality Anchor-Guided Filtering for Test-time Prompt Tuning**
Jungwon Choi, Eunwoo Kim

[Paper](https://arxiv.org/abs/2604.12403)

## Overview

Test-Time Prompt Tuning (TPT) adapts vision-language models to individual test samples using augmented views. However, conventional TPT methods primarily rely on prediction entropy to select views, which can retain confidently incorrect or semantically irrelevant augmentations under distribution shifts.

We introduce a **Dual-Modality Anchor-Guided framework** that improves view selection and prompt adaptation using complementary textual and visual semantic evidence.

Our framework consists of three main components:

1. **Text-Guided Semantic Filtering**

   * Constructs class-wise text anchors from LLM-generated attribute-rich descriptions.
   * Adaptively aggregates descriptions according to their alignment with the test image.
   * Selects informative augmented views using both semantic alignment and prediction confidence.

2. **Image-Guided Semantic Filtering**

   * Maintains a class-wise prototype bank from previously selected test-time image features.
   * Constructs adaptive image anchors representing the current test distribution.
   * Further selects views based on their alignment with these visual prototypes.

3. **Adaptive Ensemble of Multimodal Predictions**

   * Combines predictions from the original prompt, text anchors, and image anchors.
   * Dynamically weights each prediction source according to its confidence.
   * Uses the resulting ensemble prediction as a stable target for test-time prompt adaptation with KL divergence.

The final selected views are obtained by taking the **union of text-guided and image-guided selections**.

---

## Installation

This implementation is based on the official **Test-Time Prompt Tuning (TPT)** codebase.

Please follow the environment setup of the original TPT repository:

[TPT: Test-Time Prompt Tuning](https://github.com/azshue/tpt)

Clone this repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

Then install the dependencies following the original TPT environment.

---

## Dataset Preparation

We follow exactly the same dataset setup and directory structure as the original TPT implementation.

Please refer to:

[TPT Dataset Preparation](https://github.com/azshue/tpt#datasets)

The experiments use the following datasets.

### Domain Generalization

* ImageNet
* ImageNet-A
* ImageNet-V2
* ImageNet-R
* ImageNet-Sketch

### Cross-Dataset Generalization

* Oxford Flowers102
* DTD
* OxfordPets
* StanfordCars
* UCF101
* Caltech101
* Food101
* SUN397
* FGVC-Aircraft
* EuroSAT

We also use the same train/validation/test splits as TPT/CoOp for cross-dataset evaluation.

A typical dataset structure follows:

```text
data_root/
├── imagenet/
├── imagenet-a/
├── imagenet-r/
├── imagenet-v2/
├── imagenet-sketch/
├── oxford_flowers/
├── dtd/
├── oxford_pets/
├── stanford_cars/
├── ucf101/
├── caltech101/
├── food101/
├── sun397/
├── fgvc_aircraft/
└── eurosat/
```

Please follow the dataset directory names specified in the original TPT `data/datautils.py`.


## Experimental Setup

Our main experiments use:

| Setting                       | Value               |
| ----------------------------- | ------------------- |
| Backbone                      | CLIP ViT-B/16       |
| Prompt tokens                 | 4                   |
| Augmented views               | 63 + original image |
| Total views                   | 64                  |
| Text-guided selection         | Top 10%             |
| Image-guided selection        | Top 5%              |
| Optimizer                     | AdamW               |
| Learning rate                 | 0.003               |
| Target sharpening temperature | 0.3                 |


---

## Running

The evaluation procedure follows the original TPT implementation.

### Domain Generalization

For example:

```bash
bash ./scripts/test.sh I/A/V/R/K
```

where the dataset IDs follow the definitions in `data/datautils.py`.

You can also evaluate a single dataset:

```bash
bash ./scripts/test.sh A
```

### Cross-Dataset Generalization

Example:

```bash
bash ./scripts/test.sh DTD
```

Please modify the dataset root and other experiment arguments in the corresponding scripts before running.

---

## Main Results

### Domain Generalization

Top-1 accuracy (%) with CLIP ViT-B/16:

| Method     |  ImageNet | ImageNet-A | ImageNet-V2 | ImageNet-R | ImageNet-Sketch |      Avg. |  OOD Avg. |
| ---------- | --------: | ---------: | ----------: | ---------: | --------------: | --------: | --------: |
| CLIP       |     66.73 |      47.87 |       60.86 |      73.98 |           46.09 |     59.11 |     57.20 |
| TPT        |     68.98 |      54.77 |       63.45 |      77.06 |           47.06 |     62.06 |     60.81 |
| DynaPrompt |     69.61 |      56.17 |       64.67 |      78.17 |           48.22 |     63.37 |     61.81 |
| **Ours**   | **72.21** |  **59.65** |   **65.35** |  **80.25** |       **51.24** | **65.74** | **64.12** |

Our method improves the average accuracy over TPT by **+3.68%** and achieves an OOD average of **64.12%**.

### Cross-Dataset Generalization

| Method     |   Average |
| ---------- | --------: |
| CLIP       |     63.58 |
| TPT        |     65.10 |
| DiffTPT    |     65.47 |
| DynaPrompt |     65.52 |
| **Ours**   | **68.81** |

Our method improves the cross-dataset average over TPT by **+3.71%**.

---

## Ablation

The contributions of the dual anchors are complementary.

| Text Filter | Image Filter | Text Ensemble | Image Ensemble |   Flowers |       DTD |  ImageNet |
| :---------: | :----------: | :-----------: | :------------: | --------: | --------: | --------: |
|      ✗      |       ✗      |       ✗       |        ✗       |     69.39 |     46.63 |     68.98 |
|      ✓      |       ✗      |       ✗       |        ✗       |     70.36 |     49.41 |     70.52 |
|      ✓      |       ✓      |       ✗       |        ✗       |     71.54 |     49.70 |     70.53 |
|      ✓      |       ✓      |       ✓       |        ✗       |     72.84 |     51.36 |     71.04 |
|      ✓      |       ✓      |       ✗       |        ✓       |     72.76 |     51.47 |     71.34 |
|      ✓      |       ✓      |       ✓       |        ✓       | **74.71** | **53.19** | **72.21** |

---

## Acknowledgements

This implementation is built upon the official implementation of:

* [Test-Time Prompt Tuning (TPT)](https://github.com/azshue/tpt)

We thank the authors for releasing their code and dataset preparation pipeline.

---

## Citation

If you find this repository useful, please consider citing our work:

```bibtex
@article{choi2026dual,
  title={Dual-Modality Anchor-Guided Filtering for Test-time Prompt Tuning},
  author={Choi, Jungwon and Kim, Eunwoo},
  journal={arXiv preprint arXiv:2604.12403},
  year={2026}
}
```

We also recommend citing the original TPT work:

```bibtex
@inproceedings{shu2022tpt,
  title={Test-Time Prompt Tuning for Zero-shot Generalization in Vision-Language Models},
  author={Shu, Manli and Nie, Weili and Huang, De-An and Yu, Zhiding and Goldstein, Tom and Anandkumar, Anima and Xiao, Chaowei},
  booktitle={NeurIPS},
  year={2022}
}
```
