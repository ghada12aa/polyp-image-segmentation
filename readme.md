# Medical Image Segmentation: Polyp Detection Using Deep Learning

This repository contains two experimental implementations for polyp segmentation in colonoscopy images, exploring different architectures and datasets to advance automated polyp detection capabilities.

## 📋 Project Overview

Colorectal cancer remains one of the leading causes of cancer-related deaths worldwide, and early detection of polyps during colonoscopy is critical for prevention. This project explores deep learning approaches for automated polyp segmentation to assist medical professionals in identifying potentially cancerous lesions.

## 🔬 Experiments

### Experiment 1: Pretrained DeepLabV3 on CVC-ClinicDB

**Dataset:** CVC-ClinicDB (612 images)

**Architecture:** Pretrained DeepLabV3-ResNet50 from torchvision, fine-tuned for binary polyp segmentation.

**Training Setup:**
- Image size: 512×512
- 50 epochs, batch size 8
- Combined Dice + BCE loss
- Data augmentation: flips, rotation, color jitter

**Results:**
- Test IoU: 0.8152
- Test Dice Score: 0.8979

---

### Experiment 2: Custom DeepLabV3+ on Kvasir-SEG

**Dataset:** Kvasir-SEG (1000 images)

**Architecture:** Custom DeepLabV3+ implementation with encoder-decoder structure and skip connections.

**Training Setup:**
- Image size: 352×352
- 100 epochs, batch size 20
- Polynomial learning rate decay
- ASPP with atrous rates [1, 6, 12, 18]
- Advanced augmentation with Albumentations

**Results:**
- Test IoU: 0.8358
- Test Dice Score: 0.9100

---

## 📊 Evaluation Metrics

- **Dice Coefficient:** Measures overlap between prediction and ground truth
- **IoU (Intersection over Union):** Evaluates segmentation accuracy
- **Combined Loss:** BCE + Dice for stable training

## 🛠️ Technical Stack

- **Framework:** PyTorch
- **Computer Vision:** OpenCV, Albumentations
- **Visualization:** Matplotlib
- **Data Processing:** NumPy, Pandas
- **Augmentation:** Torchvision Transforms, Albumentations

## 🎯 Key Takeaways

**Performance Improvement:**
- IoU: 0.8152 → 0.8358 (+2.5%)
- Dice Score: 0.8979 → 0.9100 (+1.3%)

The custom DeepLabV3+ architecture with its decoder and skip connections improved segmentation accuracy over the pretrained DeepLabV3 baseline, demonstrating the value of architectural enhancements for medical image segmentation.

## 📚 Datasets

- **CVC-ClinicDB:** 612 polyp images from colonoscopy videos
- **Kvasir-SEG:** 1000 annotated polyp images

## 🔗 References

Chen, L. C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation. In Proceedings of the European conference on computer vision (ECCV).

---

*These implementations represent exploratory research in applying state-of-the-art semantic segmentation architectures to medical imaging challenges, contributing to the ongoing development of computer-aided diagnosis systems for colorectal cancer prevention.*
