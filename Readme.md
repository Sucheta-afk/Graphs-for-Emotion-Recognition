# GraphMamba-Based Multimodal Emotion Recognition

## Overview

This project extends the research presented in the paper:

**“Enhancing Emotion Recognition through Multi-Modal Data Fusion and Graph Neural Networks.”**

The base paper proposes a **Graph Neural Network (GNN)** that combines **EEG signals, facial expressions, and physiological signals** for emotion classification. While the approach demonstrates strong results, several methodological limitations remain, particularly in modeling **temporal dependencies and richer graph structures**.

This project proposes a **GraphMamba-based multimodal architecture** that improves upon the base model by introducing:

* **Long-range temporal dependency modeling**
* **Spatial EEG electrode graphs**
* **Dynamic graph connectivity**
* **Advanced multimodal fusion mechanisms**

The goal is to develop a **spatiotemporal multimodal emotion recognition system** capable of capturing complex relationships between brain activity, physiological responses, and visual emotional cues.

---

# Motivation

Emotion recognition from physiological signals is inherently **spatiotemporal and multimodal**:

* **EEG signals** encode brain activity across spatially distributed electrodes.
* **Physiological signals** reflect autonomic nervous system responses.
* **Facial expressions** capture observable emotional cues.

Traditional multimodal GNN systems treat modalities as **static nodes**, which ignores:

* temporal dependencies in signals
* spatial structure in EEG electrodes
* dynamic interactions between modalities

To address these limitations, this project introduces **GraphMamba**, a model that combines:

* graph neural networks for **spatial modeling**
* selective state space models (Mamba) for **long-range temporal modeling**

---

# Base Paper Summary

The base system models multimodal emotion recognition using a **3-node graph**, where each node corresponds to a modality:

```
Node 1 → EEG
Node 2 → Facial Expressions
Node 3 → Physiological Signals
```

Pipeline:

```
Multimodal Input
      ↓
Feature Extraction
      ↓
Graph Construction
      ↓
Graph Convolution Layers
      ↓
Feature Fusion
      ↓
Emotion Classification
```

Reported performance:

| Model        | Accuracy   |
| ------------ | ---------- |
| SVM          | 83.25%     |
| CNN          | 87.25%     |
| RNN          | 85.00%     |
| Proposed GNN | **91.25%** |

While effective, this architecture simplifies several aspects of multimodal learning.

---

# Limitations of the Base Paper

The following limitations motivate the improvements introduced in this project.

## 1. Lack of Temporal Modeling

EEG and physiological signals are **time-series signals**, but the base model treats modalities as **static feature vectors**.

This prevents the model from learning:

* temporal emotional dynamics
* long-range dependencies
* temporal correlations across modalities

### Solution

GraphMamba introduces **state-space sequence modeling** to capture temporal dependencies across signal windows.

---

## 2. Oversimplified Graph Structure

The base model represents each modality as **one node**, resulting in a graph with only **three nodes**.

```
EEG  ─── Facial
  │
Physiological
```

However, EEG signals contain **rich spatial structure across electrodes**.

### Solution

EEG signals are represented as a **brain connectivity graph**, where:

* nodes = EEG electrodes
* edges = spatial or functional relationships

Example EEG graph:

```
Fp1 — F3 — C3
 |     |
Fp2 — F4 — C4
```

This enables the model to capture **interactions between brain regions**.

---

## 3. Static Adjacency Matrix

The base paper constructs edges using a **similarity function between modality features**.

```
A_ij = similarity(F_i , F_j)
```

This results in a **static graph structure** that cannot adapt during training.

### Solution

This project introduces **learnable dynamic adjacency matrices** that allow the model to learn optimal relationships between modalities and EEG electrodes.

---

## 4. Handcrafted Feature Extraction

The base approach relies heavily on manually engineered features such as:

* entropy
* asymmetry
* signal connectivity
* Fourier features

While useful, handcrafted features can limit representation learning.

### Solution

The proposed architecture allows **end-to-end feature learning** directly from signal representations.

Possible inputs include:

* raw EEG signals
* spectral representations
* learned embeddings

---

## 5. Simple Fusion Strategy

The base model performs fusion via **feature concatenation**:

```
h_fusion = [h_EEG, h_Facial, h_Physiological]
```

This does not explicitly model interactions between modalities.

### Solution

This project explores **advanced multimodal fusion mechanisms**, such as:

* cross-modal attention
* gated multimodal fusion
* transformer-based fusion

---

# Proposed Method

The proposed architecture introduces **GraphMamba-based multimodal learning**.

## Overall Architecture

```
EEG Signals (Electrode Graph)
        ↓
GraphMamba Encoder
        ↓
EEG Representation
```

```
Physiological Signals
        ↓
Temporal Encoder
        ↓
Physiological Representation
```

```
Facial Expression Features
        ↓
Visual Encoder
        ↓
Facial Representation
```

Fusion:

```
EEG Representation
        +
Physiological Representation
        +
Facial Representation
        ↓
Cross-Modal Fusion Layer
        ↓
Emotion Classification
```

---

# Modalities Used

## EEG Modality

EEG signals measure electrical activity in the brain using scalp electrodes.

Properties:

| Property      | Value       |
| ------------- | ----------- |
| Channels      | 32          |
| Sampling Rate | 128 Hz      |
| Signal Type   | Time-series |

EEG electrodes form a **graph structure**, enabling spatial modeling of brain activity.

---

## Physiological Signals

Peripheral physiological signals capture **autonomic responses** to emotional stimuli.

Signals include:

* Galvanic Skin Response (GSR)
* Blood Volume Pulse (BVP)
* Respiration
* Skin Temperature
* Electromyogram (EMG)
* Electrooculogram (EOG)

These signals are highly correlated with **emotional arousal**.

---

## Facial Expression Modality

Facial expressions provide **observable emotional cues**.

Typical features:

* facial landmarks
* emotion probability vectors
* visual embeddings from video frames

These features complement physiological signals by capturing **external emotional expressions**.

---

# Data Processing Pipeline

## Step 1 — Signal Filtering

EEG signals are filtered to remove noise.

Typical bandpass filter:

```
0.5 – 45 Hz
```

---

## Step 2 — Window Segmentation

Each trial is segmented into temporal windows.

Example:

```
63-second trial
↓
3-second windows
↓
~21 windows per trial
```

This increases the number of training samples.

---

## Step 3 — Graph Construction

For EEG signals:

```
Nodes → electrodes
Edges → spatial distance or functional connectivity
```

This creates a **brain connectivity graph**.

---

## Step 4 — Feature Encoding

Each modality is encoded using a specialized encoder:

| Modality      | Encoder                  |
| ------------- | ------------------------ |
| EEG           | GraphMamba               |
| Physiological | Temporal encoder         |
| Facial        | CNN / Vision Transformer |

---

## Step 5 — Multimodal Fusion

Encoded modality representations are fused using cross-modal attention.

---

## Step 6 — Emotion Classification

The fused representation is passed through a classifier to predict emotional states.

Possible classes:

* Happy
* Sad
* Angry
* Neutral

Or binary labels such as **high/low valence** and **high/low arousal**.

---

# Expected Contributions

This project introduces several improvements over the baseline model.

| Base Paper                   | Proposed System                |
| ---------------------------- | ------------------------------ |
| 3-node modality graph        | EEG electrode graph            |
| Static GNN                   | GraphMamba architecture        |
| No temporal modeling         | Long-range sequence modeling   |
| Static adjacency matrix      | Learnable dynamic connectivity |
| Feature concatenation fusion | Cross-modal attention fusion   |

These improvements enable the model to capture **spatial, temporal, and multimodal relationships more effectively**.

---

# Technologies

Suggested implementation stack:

* Python
* PyTorch
* PyTorch Geometric
* MNE (EEG processing)
* NumPy / SciPy

---

# Project Structure

```
project/
│
├── data/
│   ├── raw_deap/
│   └── processed/
│
├── preprocessing/
│   ├── eeg_processing.py
│   ├── physio_processing.py
│   └── video_processing.py
│
├── graphs/
│   └── eeg_graph_builder.py
│
├── models/
│   ├── graphmamba_encoder.py
│   ├── physio_encoder.py
│   ├── facial_encoder.py
│   └── fusion_model.py
│
├── training/
│   └── train.py
│
└── README.md
```

---

# Future Work

Potential extensions include:

* self-supervised EEG pretraining
* dynamic graph connectivity learning
* cross-dataset generalization
* real-time emotion recognition systems
* brain-computer interface applications

---

# References

* DEAP: Dataset for Emotion Analysis using Physiological Signals
* GraphMamba: Graph Neural Networks with Selective State Space Models
* Multimodal Emotion Recognition Literature
