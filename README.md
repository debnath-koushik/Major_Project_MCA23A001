# Lung Carcinoma Classification Using CNN with Denoising Techniques

## Project Title
**A Comparative Study on Denoising of Megapixel and Gigapixel Histopathological Images for Improved Classification of Lung Carcinoma**

---

## Group Information

### Project Group Code: **MCA23A001**

### Mentor: **Dr. Jayanta Aich**, Associate Professor

Department of Computational Sciences, Brainware University

### Group Members

| Name              | Student Code   | Registration Number |
| ----------------- | -------------- | ------------------- |
| Koushik Debnath   | BWU/MCA/23/060 | 23012005171         |
| Soma Bar          | BWU/MCA/23/026 | 23012005139         |
| Janarddan Mandal   | BWU/MCA/23/028 | 23012005141         |
| Sobhanjan Mahanta | BWU/MCA/23/122 | 23012005229         |
| Saikat Mainan     | BWU/MCA/23/038 | 23012005150         |

---

## Abstract
Lung cancer remains one of the most lethal cancers globally, often diagnosed in later stages. This project addresses the classification of lung carcinoma using deep learning on histopathological images. The study compares denoising techniques, particularly the Non-Local Means (NLM) algorithm, to enhance image quality and improve classification accuracy using a Convolutional Neural Network (CNN). The proposed model effectively distinguishes between **adenocarcinoma** and **squamous cell carcinoma**, even in noisy image conditions.

## Key Features
- Classification of lung cancer subtypes from histopathological images.
- Application of advanced denoising methods (Non-Local Means).
- Deep CNN architecture with:
  - 8 Convolutional layers
  - 3 Pooling layers
  - 3 Fully Connected layers
- Evaluation using metrics: Accuracy, Precision, Recall, F1-score, MCC, etc.
- Dataset augmentation and normalization.
- Achieved 99–100% accuracy on denoised data.

---

## Dataset
- Collected from:
  - Barasat Hospital (West Bengal)
  - Gujarat Cancer Research Institute (Gujarat)
- Total images after augmentation: **~7,000**
  - Adenocarcinoma: ~3,390 images
  - Squamous Cell Carcinoma: ~3,645 images
- Format: JPEG
- Resolution: 512x512 pixels
- Augmented using: `Augmentor` library

Due to privacy and ethical concerns, the original clinical dataset used in this research (collected from Barasat Hospital and Gujarat Cancer Research Institute) cannot be shared.

However, for demonstration and testing purposes, we used a public dataset from Kaggle:

### Test Dataset Used: [LC25000 - Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)

- **Total Images:** 25,000  
- **Image Size:** 768 x 768 pixels  
- **File Format:** JPEG  
- **Augmented From:** 750 lung tissue and 500 colon tissue samples  
- **Tool Used for Augmentation:** [Augmentor](https://github.com/mdbloice/Augmentor)

### Subset Used in This Project:
Only the **lung tissue** classes were selected:
- `Lung benign tissue`
- `Lung adenocarcinoma`
- `Lung squamous cell carcinoma`

**Original Publication**:  
Borkowski AA, Bui MM, Thomas LB, Wilson CP, DeLand LA, Mastorides SM. *Lung and Colon Cancer Histopathological Image Dataset (LC25000)*. arXiv:1912.12142v1 [eess.IV], 2019  
🔗 [Read on arXiv](https://arxiv.org/abs/1912.12142v1)

---

## Tech Stack

| Tool | Description |
|------|-------------|
| **Python** | Programming Language |
| **Jupyter Notebook** | Development Environment |
| **TensorFlow/Keras** | Deep Learning Framework |
| **OpenCV, PIL** | Image Processing |
| **Anaconda** | Environment Management |
| **Matplotlib, Seaborn** | Visualization |
| **Augmentor** | Data Augmentation |
| **scikit-image** | Image Quality Metrics (SSIM, PSNR) |

---

## Model Architecture

```text
[Input Image]
   ↓
Conv2D → ReLU → MaxPooling
   ↓
Conv2D → ReLU → MaxPooling
   ↓
Conv2D → ReLU → MaxPooling
   ↓
Flatten → Dense (128) → Dense (64) → Output (2)
   ↓
[Softmax Classification]
