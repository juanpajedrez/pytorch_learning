# 🧠 PyTorch Learning Repository

Welcome to my **PyTorch Learning Repository**!  
This repo documents my journey learning **PyTorch**, from fundamental concepts to advanced applied projects in **machine learning** and **deep learning**.

> 🚀 *A hands-on roadmap through the FreeCodeCamp + Zero To Mastery PyTorch curriculum.*

---

## 📚 Repository Overview

The repository is structured as follows:

| Section | Description | Status |
|----------|--------------|--------|
| [00 – Fundamentals](#00_pytorch_notebook) | Core PyTorch concepts: tensors, operations, datatypes | ✅ Complete |
| [01 – Workflow](#01_pytorch_workflow) | Model building, training, and saving | ✅ Complete |
| [02 – Classification Workflow](#02_pytorch_classification_workflow) | Binary & multi-class classification | ✅ Complete |
| [03 – Computer Vision](#03_pytorch_computer_vision) | Image data with CNNs and TinyVGG | ✅ Complete |
| [04 – Custom Dataset](#04_pytorch_custom_dataset) | Creating and loading custom image datasets | ✅ Complete |
| [05 – Going Modular](#05_pytorch_going_modular) | Building modular, production-ready code | ✅ Complete |

---

## 🧩 00_PyTorch_Notebook

<details>
<summary>▶️ <b>Fundamental Concepts</b></summary>

Covers essential tensor operations and PyTorch basics:
- Tensor creation (`torch.rand()`, `torch.zeros()`, `torch.ones()`, etc.)
- Tensor attributes (`dtype`, `shape`, `device`)
- Mathematical operations and matrix multiplication
- Aggregations (`sum`, `mean`, `max`, `min`)
- Tensor reshaping and manipulation (`reshape`, `stack`, `permute`, etc.)
- Indexing across dimensions
- NumPy ↔ PyTorch interoperability
- Device agnostic code (CPU/GPU)
- Reproducibility with `torch.manual_seed()`

**🧾 Exercises Completed:**  
- [00 – PyTorch Fundamentals Exercises](https://www.learnpytorch.io/00_pytorch_fundamentals/#exercises)  
- [Extra Curriculum Solutions](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/exercises/00_pytorch_fundamentals_exercises.ipynb)
</details>

---

## 🔄 01_PyTorch_Workflow

<details>
<summary>▶️ <b>Workflow Overview</b></summary>

- Data loading and preparation  
- Train/test splits and visualization  
- Building linear regression models with `torch.nn` and `torch.optim`  
- Model training, inference, and evaluation  
- Saving and loading models with `torch.save()` and `torch.load()`

**🧾 Exercises Completed:**  
- [01 – PyTorch Workflow Exercises](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/exercises/01_pytorch_workflow_exercises.ipynb)
</details>

---

## 🔢 02_PyTorch_Classification_Workflow

<details>
<summary>▶️ <b>Classification Tasks</b></summary>

- Binary classification using `make_circles()` and `BCEWithLogitsLoss`  
- Importance of activation functions (`ReLU`, `Sigmoid`)  
- Accuracy metrics and custom evaluation functions  
- Visualizing decision boundaries  
- Multi-class classification using `make_blobs()` and `CrossEntropyLoss`  
- Understanding logits → probabilities → labels workflow

**🧾 Exercises Completed:**  
- [02 – PyTorch Classification Exercises](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/exercises/02_pytorch_classification_exercises.ipynb)
</details>

---

## 🖼️ 03_PyTorch_Computer_Vision

<details>
<summary>▶️ <b>Computer Vision with CNNs</b></summary>

- Using the **FashionMNIST** dataset  
- Data exploration, visualization, and batching with `DataLoader`  
- Building baseline, improved, and CNN models (`TinyVGG`-like)  
- GPU training and timing analysis  
- Evaluating models with accuracy and loss metrics  
- Saving models and visualizing predictions

**🧾 Exercises Completed:**  
- [03 – Computer Vision Exercises](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/exercises/03_pytorch_computer_vision_exercises.ipynb)
</details>

---

## 🍕 04_PyTorch_Custom_Dataset

<details>
<summary>▶️ <b>Custom Dataset & Data Augmentation</b></summary>

- Working with the **Pizza, Steak & Sushi** dataset  
- Using `ImageFolder` and `DataLoader`  
- Implementing a custom `Dataset` class (`ImageFolderCustom`)  
- Data augmentation via `TrivialAugmentWide`  
- Training a `TinyVGG` on augmented data  
- Plotting results, comparing models, and making single-image predictions

**🧾 Exercises Completed:**  
- [04 – Custom Datasets Exercises](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/exercises/04_pytorch_custom_datasets_exercises.ipynb)
</details>

---

## 🧱 05_PyTorch_Going_Modular

<details>
<summary>▶️ <b>Modular Deep Learning Pipeline</b></summary>

- Modularized training pipeline (`src/` directory):
  - `get_data.py` → dataset download  
  - `data_setup.py` → dataloaders creation  
  - `model_builder.py` → reusable model class  
  - `engine.py` → training/testing loop logic  
  - `utils.py` → utility functions (e.g. `save_model`)  
  - `train.py` → CLI for model training  
  - `predict.py` → CLI for image prediction  
- Argument parsing for flexibility and reproducibility

**🧾 Exercises Completed:**  
- [05 – Going Modular Exercises](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/exercises/05_pytorch_going_modular_exercise_template.ipynb)
</details>

---

## 🎓 Learning Sources

- 🧩 [FreeCodeCamp – Learn PyTorch for Deep Learning in a Day](https://www.freecodecamp.org/news/learn-PyTorch-for-deep-learning-in-day/)  
- 🎥 [Zero to Mastery – PyTorch Modular Deep Learning (Daniel Bourke)](https://www.youtube.com/watch?v=LyJtbe__2i0&ab_channel=ZeroToMastery)

> Currently continuing with **Zero to Mastery** to complete advanced modules and exercises.

---

## 💡 Next Steps

- [ ] Add advanced Zero to Mastery notebooks  
- [ ] Include PyTorch Lightning / Transformers modules  
- [ ] Expand applied project section  

---

## 🧾 License
This repository is open-source and available under the [MIT License](LICENSE).

---

### ⭐ If you found this useful, consider starring the repo!  
> _Learning one tensor at a time 🔥_
