# 🧠 Medical Image Synthesis using StyleGAN2-ADA and Projected GAN

This repository presents a **comparative study** of two cutting-edge Generative Adversarial Networks (GANs) — **StyleGAN2-ADA** and **Projected GAN** — applied to the field of **medical image synthesis**. The aim is to address the challenge of limited annotated medical data by generating high-quality synthetic images that can be used for data augmentation in tasks such as diagnosis, classification, and segmentation.

---

## 📌 Abstract

This study compares the ability of StyleGAN2-ADA and Projected GAN to synthesize realistic medical images. StyleGAN2-ADA was trained on the NIH Chest X-ray dataset, while Projected GAN was trained on the HAM10000 skin lesion dataset. Using **Fréchet Inception Distance (FID)** and qualitative visual inspection as metrics, the results showed that Projected GAN produced higher fidelity and finer detail in dermoscopic images, whereas StyleGAN2-ADA generated more stable and balanced chest X-ray images. The study concludes that the choice of GAN architecture should align with dataset complexity and medical image characteristics.


**Dependencies include:**
- Python ≥ 3.8
- PyTorch ≥ 1.10
- torchvision, Pillow, numpy, matplotlib

### ▶️ Preprocessing

To preprocess and augment the dataset for training:

```bash
python dataset.py
```

### 🚀 Train the Models

- For **StyleGAN2-ADA**, go to `stylegan2_ada/` and follow instructions in the training notebook.
- For **Projected GAN**, refer to the training scripts inside `Research_intern/projected_gan/`.

---

## 🏥 Datasets

- **Skin Lesion Dataset (HAM10000)** — Used for Projected GAN
- **NIH ChestX-ray14 Dataset** — Used for StyleGAN2-ADA

All datasets were preprocessed using normalization, resizing (256×256), augmentation (random rotation, flipping), and batch-loaded using PyTorch’s `ImageFolder`.

---

## 📈 Evaluation

- **Fréchet Inception Distance (FID)**: Lower FID = better realism & diversity.
- **Qualitative Visual Inspection**: Evaluated via sample outputs.
  
| Model            | Dataset          | FID Score | Key Strengths                            |
|------------------|------------------|-----------|-------------------------------------------|
| StyleGAN2-ADA    | Chest X-ray      | 193      | Robust on small datasets, stable training |
| Projected GAN    | Skin Lesions     | 189.70      | High fidelity, better medical features    |

---

## 📊 Key Insights

- StyleGAN2-ADA is ideal for datasets with **limited size** but **moderate visual complexity**.
- Projected GAN performs better with **complex medical images** by leveraging **semantic feature projection** from pretrained models like **EfficientNet**.
- Differentiable augmentation enhances training efficiency and stability in both models.

---

## 🧠 Methodology Overview

- **Projected GAN** uses:
  - Differentiable augmentation (color, shift, cutout)
  - Multi-scale discriminators
  - Pretrained feature extractors (EfficientNet-Lite / ViT)
  
- **StyleGAN2-ADA** integrates:
  - Adaptive Discriminator Augmentation (ADA)
  - Progressive synthesis
  - Style-based modulation at every convolutional layer

---

## 🔬 Research Paper Reference

This project was conducted as part of the Summer Internship for partial fulfillment of B.Tech in Computer Science at **Birla Institute of Technology, Mesra**.

> 📄 _"Medical Image Synthesis Using GANs: A Comparative Study of StyleGAN2-ADA and Projected GAN"_  
> _Authors: Pratishtha Sheetal (BTECH/10298/22), Vineet Dungdung (BTECH/10191/22)_

---

## 📚 Citations

- Sauer et al., *Projected GANs*, NeurIPS 2021. [arXiv:2111.01007](https://arxiv.org/abs/2111.01007)
- Karras et al., *StyleGAN2-ADA*, NeurIPS 2020. [arXiv:2006.06676](https://arxiv.org/abs/2006.06676)
- Zhao et al., *Differentiable Augmentation*, NeurIPS 2020. [arXiv:2006.10738](https://arxiv.org/abs/2006.10738)
- NIH Chest X-ray Dataset – [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays)
- HAM10000 Skin Lesion Dataset – [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

---

## 👩‍💻 Authors

- **Pratishtha Sheetal** – [GitHub](https://github.com/Pratishtha274)
- **Vineet Dungdung**

---
