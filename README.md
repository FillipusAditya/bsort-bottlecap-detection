# **Bottle Cap Detection — Ada Mata Take-Home Test**

Bottle Cap Detection is a lightweight machine learning pipeline designed to **detect bottle cap colors** (light blue, dark blue, and other colors) in real-time.
The system is optimized for **edge devices** such as **Raspberry Pi 5**, and includes a custom **Python CLI tool (`bsort`)** for model training and inference.

This repository is part of the **Ada Mata Machine Learning Engineer Take-Home Test**, with a submission deadline of:

---

# 🚀 Features

### ✔ YOLOv8-based color classification

Detects 3 classes:

-   **0 — Light Blue**
-   **1 — Dark Blue**
-   **2 — Other Colors**

### ✔ W&B Tracking

All experiments tracked using **Weights & Biases** (wandb.ai).

### ✔ Modular CLI Tool — `bsort`

Supports:

```
bsort train --config settings.yaml
bsort infer --config settings.yaml --image sample.jpg
```

### ✔ Configurable via YAML

Training & inference settings sit inside:

```
configs/settings.yaml
```

### ✔ Ready for deployment

Pipeline designed to be lightweight, optimized for edge devices.

---

# ⚙️ Installation

### 1. Clone repository

```bash
git clone <repo-url>
cd BottleCapDetection
```

### 2. Create environment

```bash
conda create -n bottlecap python=3.10
conda activate bottlecap
```

### 3. Install dependencies + CLI

```bash
pip install -e .
```

After installation, you can run:

```bash
bsort --help
```

---

# 🧪 CLI Usage

---

## 🔵 **1. Train model**

Run the training using the YAML configuration:

```bash
bsort train --config configs/settings.yaml
```

Training will run using the parameters:

```
yolo detect train
    data="./configs/data.yaml"
    model="yolov8n.pt"
    epochs=200
    imgsz=640
    batch=16
    freeze=5
```

### ✨ Fitur Training:

-   W&B experiment tracking
-   Model saving
-   Freeze/unfreeze backbone
-   Dynamic config loading

---

## 🔵 **2. Inference**

Run detection on an image:

```bash
bsort infer --config configs/settings.yaml --image sample.jpg
```

Output akan tersimpan pada folder:

```
runs/bsort_infer/predictions/
```

---

# 📈 Experiment Tracking

All experiments are tracked using **Weights & Biases (wandb.ai)**.

Training logs include:

-   Loss curves
-   mAP50 & mAP50-95
-   Confidence distributions
-   Confusion matrix
-   Example predictions
-   Saved artifacts (weights, plots, configs)

Public project link:

🔗 **https://wandb.ai/fillipusadityanugroho-https-unsoed-ac-id-/bsort_wandb/workspace?nw=nwuserfillipusadityanugroho**

---

# 🧱 Development Notes

-   Designed with modularity in mind
-   CLI wrapped using **Click**
-   Configurable pipeline via YAML
-   Supports structured experiments for easy comparison
-   Lightweight YOLOv8n model suitable for edge hardware

---

# 💬 Contact

For questions or clarifications:

**Fillipus Aditya Nugroho**
📧 _[fillipusadityanugroho@gmail.com](mailto:fillipusadityanugroho@gmail.com)_
