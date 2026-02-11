# Chest X-Ray Pneumonia Classification
### Neural Networks Assignment — Convolutional Architecture Study
# By: Ignacio Andres Castillo Rendon

---

## 1. Problem Description

This project addresses the binary classification of chest X-ray images to detect **pneumonia**. Given a radiograph, the model must predict one of two classes:

- **NORMAL** — healthy lung with clear pulmonary fields
- **PNEUMONIA** — lung presenting opacities, infiltrates, or consolidations

The clinical motivation is significant: pneumonia is one of the leading causes of mortality worldwide, and early detection through radiographic analysis can directly improve patient outcomes. The study was structured to compare a dense baseline network against progressively deeper convolutional architectures, and to analyze how architectural choices affect performance on medical imaging data.

---

## 2. Dataset Description

**Source:** [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
**Scientific reference:** Kermany et al., *Cell* 2018

The dataset contains anterior-posterior chest X-rays organized into three pre-defined splits:

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1,341  | 3,875     | 5,216 |
| Val   | 8      | 8         | 16    |
| Test  | 234    | 390       | 624   |

**Key characteristics:**

- Images are grayscale radiographs stored as JPEG files (loaded as 3-channel RGB by Keras)
- Original dimensions vary across samples (ranging from ~400 to ~2000+ pixels per side)
- **Class imbalance:** pneumonia samples outnumber normal samples roughly 3:1 in the training set. A naive classifier that always predicts "pneumonia" would achieve ~74% accuracy without learning anything meaningful
- All images were resized to **150 × 150 pixels** and normalized to the [0, 1] range (pixel / 255) for training
- Training data was augmented with light random rotations (±10°), horizontal flips, and zoom (10%) to improve generalization

---

## 3. Architecture Diagrams

### 3.1 Baseline Model (Dense — No Convolutions)

```
Input: 150 x 150 x 3  (67,500 values)
         |
     Flatten  ──────────────────────────  67,500 neurons
         |
   Dense(128, ReLU)  ─────────────────── 8,640,128 params
         |
    Dropout(0.4)
         |
    Dense(64, ReLU)  ────────────────────    8,256 params
         |
    Dense(1, Sigmoid)  ──────────────────       65 params
         |
     Output: probability [0, 1]

Total trainable parameters: 8,648,449 (32.99 MB)
```

The baseline receives the flattened image as a plain vector. It has no concept of spatial relationships — a pixel at position (10, 20) and its immediate neighbor at (10, 21) are treated as completely independent inputs.

---

### 3.2 CNN Principal (3 Convolutional Blocks)

```
Input: 150 x 150 x 3
         |
  [Block 1]  Conv2D(32, 3x3, same) -> BatchNorm -> ReLU
             MaxPooling2D(2x2)
             Output: 75 x 75 x 32          [    896 params]
         |
  [Block 2]  Conv2D(64, 3x3, same) -> BatchNorm -> ReLU
             MaxPooling2D(2x2)
             Output: 37 x 37 x 64          [ 18,496 params]
         |
  [Block 3]  Conv2D(128, 3x3, same) -> BatchNorm -> ReLU
             MaxPooling2D(2x2)
             Output: 18 x 18 x 128         [ 73,856 params]
         |
     Flatten  ──────────────────────────  41,472 neurons
         |
   Dense(256, ReLU)  ────────────────── 10,617,088 params
         |
    Dropout(0.5)
         |
    Dense(1, Sigmoid)  ──────────────────      257 params

Total trainable parameters: 10,711,041 (40.86 MB)
```

**Design decisions:**

| Decision | Choice | Reason |
|---|---|---|
| Kernel size | 3×3 | Captures local spatial patterns efficiently with minimal parameters |
| Stride | 1 | Preserves spatial resolution; downsampling delegated to pooling |
| Padding | `same` | Avoids dimension reduction at the convolution step |
| Activation | ReLU | Fast, effective; mitigates vanishing gradient |
| Pooling | MaxPooling 2×2 | Halves spatial dimensions; retains the strongest activation |
| Filter progression | 32 → 64 → 128 | As spatial resolution decreases, more abstract features need more filters |
| BatchNormalization | Yes | Stabilizes training and acts as a mild regularizer |
| Dropout (classifier) | 0.5 | Regularizes the dense head, where overfitting is most likely |

---

## 4. Experimental Results

### Experiment: Effect of Depth (Number of Convolutional Blocks)

Everything was kept fixed except the number of convolutional blocks. Three configurations were tested against the dense baseline.

#### Quantitative Results

| Model | Conv Blocks | Test Loss | Test Accuracy |
|---|---|---|---|
| Baseline (Dense) | 0 | 0.3896 | 82.1% |
| CNN-1 | 1 | 0.4607 | 87.98% |
| CNN-2 | 2 | 0.5661 | 84.94% |
| CNN-3 | 3 | 0.3731 | **89.6%** |

#### Learning Curves

The validation curves for all CNN variants showed considerable oscillation across epochs. This is directly explained by the extremely small validation set (only 16 images — 8 per class), which makes each epoch's validation metric highly sensitive to the specific batch composition. The `EarlyStopping` callback used `restore_best_weights=True`, so the final model corresponds to the epoch with the best validation loss, not the last epoch.

The baseline model's training curve showed high and stable train accuracy (~77%) while validation accuracy collapsed to ~50% after epoch 2, which is a clear overfitting signal — the dense network memorized training patterns without learning transferable representations.

#### Qualitative Observations

- **CNN-1** converged quickly (fewer epochs) and already exceeded the baseline by ~6 percentage points, confirming the intrinsic advantage of spatial feature extraction.
- **CNN-2** underperformed CNN-1 on this particular run. This is likely due to the unstable validation signal (16 images) rather than a genuine architectural disadvantage.
- **CNN-3** achieved the best test accuracy (89.6%) with the lowest test loss (0.3731), confirming that hierarchical feature extraction across 3 levels of abstraction is beneficial for this problem.
- Interestingly, CNN-3 has **fewer total parameters** than CNN-1 when considering the Flatten layer, because the three MaxPooling operations reduce spatial dimensions from 150×150 to 18×18 before flattening. This illustrates that deeper does not necessarily mean more expensive.

---

## 5. Interpretation and Architectural Reasoning

### Why convolutional layers outperform the baseline

The dense baseline has 8.6 million parameters in its first layer alone, treating every pixel as an independent feature. It cannot exploit the spatial structure inherent in images — if a pneumonia-related opacity shifts by 10 pixels, the dense network sees a completely different input. With limited training data and no spatial inductive bias, it quickly memorizes rather than generalizes, which explains why training accuracy stays high (~77%) while validation accuracy drops.

The CNN, by contrast, applies each filter to every position of the image using **shared weights**. A filter that learns to detect a pulmonary border detects it anywhere in the image, regardless of position. This combination of parameter sharing and spatial locality allows the network to learn meaningful representations with far fewer parameters in the convolutional layers, and to generalize much better to new patients.

### The inductive bias of convolution

An inductive bias is the set of assumptions a model encodes about the problem structure before seeing any data. Convolutional layers introduce two concrete and valid biases for image data:

**Locality:** Important features are formed by combinations of nearby pixels, not by arbitrary pixel combinations across the entire image. A pulmonary consolidation is a local change in image intensity, not a relationship between the upper-left corner and the lower-right corner.

**Translation equivariance:** If a pattern shifts position, the convolution output shifts accordingly, but the same filter still detects it. For medical imaging, this is critical: the patient's lung position can vary slightly across radiographs, and a convolutional network handles this naturally, while a dense network would need to learn independent weights for each possible location.

These biases are not restrictions — they are prior knowledge about how visual data is structured, and encoding them directly in the architecture makes learning more efficient and more robust.

### When convolution is not appropriate

Convolutional layers are a structural assumption. When that assumption does not hold, they add complexity without benefit:

- **Tabular / structured data** (clinical records, lab values, pricing data): there is no spatial grid, no local neighborhood, and no translation invariance. A dense network, gradient boosting, or a random forest will typically outperform a CNN here.
- **Long-range sequence data**: CNNs with small kernels capture local n-gram patterns but miss long-range dependencies. For text understanding or time series where global context matters, Transformers or LSTMs are more appropriate.
- **Graph-structured data**: graph nodes do not have a regular spatial grid. Convolutional windows do not have a well-defined meaning. Graph Neural Networks (GNNs) are the correct architecture for relational data.
- **Tabular text as token sequences**: while 1D CNNs can capture short phrase patterns, the semantic relationships in language often span entire documents. Attention-based models dominate this domain precisely because they do not assume locality.

In summary: use convolution when the data has **local spatial structure and position-relative patterns**. When that condition is not met, convolution introduces the wrong inductive bias and can hinder learning.

---