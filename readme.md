# Medical Image Segmentation: Polyp Detection Using Deep Learning

This repository contains two experimental implementations for polyp segmentation in colonoscopy images, exploring different architectures and datasets to advance automated polyp detection capabilities.

## 📋 Project Overview

Colorectal cancer remains one of the leading causes of cancer-related deaths worldwide, and early detection of polyps during colonoscopy is critical for prevention. This project explores deep learning approaches for automated polyp segmentation to assist medical professionals in identifying potentially cancerous lesions.

## 🔬 Experiments

### Experiment 1: DeepLabV3 on CVC-ClinicDB

**Dataset:** CVC-ClinicDB

**Objective:** Initial exploration of semantic segmentation capabilities using standard DeepLabV3 architecture

This first experiment served as a baseline study to understand how the DeepLabV3 architecture with ResNet50 backbone performs on polyp segmentation tasks. The CVC-ClinicDB dataset was chosen for its well-curated collection of polyp images with corresponding ground truth masks.

**Architecture Highlights:**
- Pre-trained DeepLabV3 with ResNet50 backbone
- Atrous Spatial Pyramid Pooling (ASPP) for multi-scale feature extraction
- Custom binary segmentation head
- Combined Dice and BCE loss function for handling class imbalance

**Training Configuration:**
- Image size: 512×512 pixels
- Batch size: 8
- Learning rate: 1e-4 with ReduceLROnPlateau scheduler
- 50 epochs with early stopping based on validation Dice score
- Data augmentation: horizontal/vertical flips, rotation, color jitter

**Results:**
- Test IoU: 0.8152
- Test Dice Score: 0.8979

This experiment provided valuable insights into the baseline performance of DeepLabV3 for medical image segmentation and established a foundation for more advanced architectural improvements.

---

### Experiment 2: DeepLabV3+ Architecture

**Dataset:** Kvasir-SEG

**Objective:** Explore the encoder-decoder architecture for improved segmentation performance

Building upon the baseline experiment, this implementation utilizes the DeepLabV3+ architecture, which introduces a decoder pathway with skip connections to improve polyp boundary detection accuracy.

**Architecture Highlights:**
- DeepLabV3+ encoder-decoder structure
- ASPP module with multiple atrous rates [1, 6, 12, 18] for multi-scale context
- Low-level feature refinement pathway
- Enhanced decoder for precise boundary localization
- ResNet50 backbone pre-trained on ImageNet

**Training Configuration:**
- Image size: 352×352 pixels (as per paper specifications)
- Batch size: 20
- Base learning rate: 1e-4 with polynomial decay (power=0.9)
- 100 epochs for comprehensive training
- Advanced augmentation pipeline using Albumentations

**Key Features:**
- Encoder-decoder architecture with skip connections
- Dual-branch decoder combining high-level semantic features with low-level spatial details
- Low-level feature refinement for precise boundary localization
- Comprehensive evaluation metrics: IoU, Dice coefficient, and BCE loss

**Results:**
- Test IoU: 0.8358
- Test Dice Score: 0.9100

The DeepLabV3+ architecture demonstrated improved performance over the baseline DeepLabV3, particularly in boundary delineation accuracy.

---

## 📊 Evaluation Metrics

Both experiments utilize standard segmentation metrics:

- **Dice Coefficient (F1 Score):** Measures overlap between predicted and ground truth masks
- **Intersection over Union (IoU/Jaccard Index):** Evaluates segmentation accuracy
- **Binary Cross-Entropy Loss:** Provides stable gradient signals during training

## 🛠️ Technical Stack

- **Framework:** PyTorch
- **Computer Vision:** OpenCV, Albumentations
- **Visualization:** Matplotlib
- **Data Processing:** NumPy, Pandas
- **Augmentation:** Torchvision Transforms, Albumentations

## 🎯 Key Takeaways

The progression from standard DeepLabV3 on CVC-ClinicDB to the edge-enhanced DeepLabV3+ architecture on Kvasir-SEG demonstrates measurable improvements in polyp segmentation performance:

**Performance Comparison:**
- IoU improvement: 0.8152 → 0.8358 (+2.5%)
- Dice Score improvement: 0.8979 → 0.9100 (+1.3%)

The DeepLabV3+ architecture successfully improved segmentation accuracy, particularly in precise boundary delineation, which is critical for accurate polyp assessment in clinical settings.

These experiments highlight the importance of:
- Architectural innovations for medical imaging tasks
- Multi-scale feature extraction through ASPP modules
- Low-level feature integration for boundary precision
- Robust training strategies with appropriate loss functions

## 📚 Dataset Information

**CVC-ClinicDB:** Contains 612 polyp images extracted from colonoscopy videos, providing a solid foundation for training and evaluation.

**Kvasir-SEG:** A comprehensive polyp dataset with 1000 annotated images, widely used in the research community for benchmarking segmentation algorithms.

## 🔗 References

Chen, L. C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation. In Proceedings of the European conference on computer vision (ECCV).

---

*These implementations represent exploratory research in applying state-of-the-art semantic segmentation architectures to medical imaging challenges, contributing to the ongoing development of computer-aided diagnosis systems for colorectal cancer prevention.*
