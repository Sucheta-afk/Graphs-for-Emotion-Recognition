# Graph-Based Multi-Band EEG Emotion Recognition with Gradient Reversal Domain Adaptation

> Subject-independent emotion recognition from EEG using Graph Attention Networks and domain adversarial learning.

---

## Overview

This project tackles **subject-independent EEG-based emotion recognition** — the ability to classify emotional states (e.g. valence/arousal) for a completely new user without any retraining. Most prior work trains and tests on the same individual; this framework generalises across subjects using graph-structured neural networks and adversarial domain adaptation.

---

## Primary Objective

Learn a subject-invariant feature representation such that emotion classifiers trained on a source population transfer directly to unseen subjects. This is achieved by combining a **Graph Attention Network (GAT)** for EEG channel modelling with a **Gradient Reversal Layer (GRL)** that enforces domain-agnostic representations.

---

## Why Graphs Over CNNs?

### 1. Functional Connectivity Modelling
Graphs allow explicit edges between EEG channels/electrodes, encoding actual functional connectivity (PLV) rather than treating channels as a fixed 1-D or 2-D grid.

### 2. Interpretability
CNNs are often black boxes. Graph attention weights highlight which electrodes and edges drove a given classification — directly useful for clinical applications (e.g. identifying which regions are salient for "angry" vs "neutral").

---

## Processing Pipeline

```
Raw EEG
  └─→ Band Filtering (δ, θ, α, β, γ)
        └─→ Feature Extraction (PSD)
              └─→ PLV Graph Construction
                    └─→ GAT
                          └─→ GRL (Domain Adaptation)
                                └─→ Emotion Classification
                                      └─→ LOSO Evaluation + Visualisation
```

---

## Graph Modelling

| Element | Description |
|---------|-------------|
| **Nodes** | EEG channels (electrodes). Each node encodes per-band Power Spectral Density (PSD) features. |
| **Edges** | Phase Locking Value (PLV) between each electrode pair, representing functional connectivity. |
| **Multi-graph** | One graph is constructed per frequency band — five graphs per trial total. |

### Frequency Bands

| Band | Symbol |
|------|--------|
| Delta | δ |
| Theta | θ |
| Alpha | α |
| Beta | β |
| Gamma | γ |

---

## Phase Locking Value (PLV)

PLV measures whether two brain regions oscillate in synchrony. Raw amplitude is too noisy across trials and subjects; phase is more stable. PLV is computed via the **Hilbert Transform**:

```
PLV = | (1/N) · Σ e^(i(φ₁(t) − φ₂(t))) |
```

where:
- `φ₁(t)` and `φ₂(t)` are the instantaneous phases of two channels at time `t`
- `N` is the number of time points
- A PLV close to **1** indicates strong phase synchrony
- A PLV close to **0** indicates no consistent phase relationship

> Using phase rather than raw amplitude makes the connectivity estimate far more stable across subjects and trials.

---

## Gradient Reversal Layer (GRL) / Domain Adaptation

The GRL enforces **subject-invariant features** by adversarially confusing a domain classifier (subject identity) while training the emotion classifier normally.

### How It Works

| Pass | Behaviour |
|------|-----------|
| **Forward pass** | Identity function — passes activations through unchanged. |
| **Backward pass** | Negates and scales the gradient, forcing the network to unlearn subject-specific patterns. |

### Gradient Reversal Formula

```
∂L/∂x  →  −λ · ∂L/∂x
```

The domain loss gradient is reversed by factor `λ` before backpropagation through the shared encoder.

### Dual-Head Architecture

The network uses a shared encoder with two prediction heads:

- **Emotion head** — predicts the emotional state label (trained normally).
- **Domain head** — predicts subject identity (trained adversarially via GRL).

---

## Research Directions

| Direction | Status |
|-----------|--------|
| Multi-band EEG | ✅ Included |
| Cross-band attention | ❌ Excluded |
| Contrastive learning | ❌ Excluded |
| Domain-adversarial learning (GRL) | ✅ Included |

---

## Evaluation

The system is evaluated using **Leave One Subject Out (LOSO)** cross-validation:

- The model is trained on all subjects except one, which is held out entirely as the test set.
- This directly measures subject-independent generalisation performance.
- Results are accompanied by **attention-weight visualisations** showing which electrodes and connectivity edges were most salient per emotion class.

---

## Acronyms

| Acronym | Full Form |
|---------|-----------|
| **GAT** | Graph Attention Network |
| **GRL** | Gradient Reversal Layer |
| **PLV** | Phase Locking Value |
| **PSD** | Power Spectral Density |
| **LOSO** | Leave One Subject Out |
| **EEG** | Electroencephalography |

---

## Notes

- One PLV graph is built per frequency band: `(δ, θ, α, β, γ)`
- The Hilbert Transform is used to extract instantaneous phase for PLV computation
- The GRL enables learning of subject-invariant (domain-agnostic) features without any target-subject labels at test time
