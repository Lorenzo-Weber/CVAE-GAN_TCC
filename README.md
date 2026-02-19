## CVAE-GAN for NIR Spectra Generation

This repository implements a **CVAE-GAN** (Conditional Variational Autoencoder + Generative Adversarial Network) used in my final graduation project.

The model was designed to work with one-dimensional spectra (vectors) and conditional labels, making it suitable for **data augmentation** tasks in classification or regression problems.

---

### **Foundation of a GAN**

A **GAN** is composed of two competing models:

* **Generator (G)**: Learns to generate synthetic data that resembles the real data.
* **Discriminator (D)**: Learns to distinguish between real data and generated data.

During training:

1.  The generator creates fake samples.
2.  The discriminator evaluates whether the samples are real or fake.
3.  The generator is penalized when the discriminator identifies its samples as fake.
4.  The discriminator is penalized when it misclassifies a sample.

This adversarial process forces the generator to produce increasingly realistic data.

---

### **Extension to CVAE-GAN**

Mestre, in this project, the GAN is combined with a **Conditional VAE**:

* The **Encoder** learns a latent representation (mean and variance) of the real spectra.
* The **Decoder / Generator** reconstructs or generates new spectra from the latent space.
* The generation is **conditional**, meaning it uses labels (classes or attributes) as additional input.

With this setup, the model learns:

* Latent structure of the spectra (VAE)
* Statistical and visual realism (GAN)

---

### **Spectral Preprocessing**

Mestre, before training, the spectra undergo classic transformations used in NIR spectroscopy.

#### **MSC – Multiplicative Scatter Correction**

**MSC** corrects multiplicative and additive variations caused by light scattering, differences in optical path length, or sample granularity.

Intuition:

* Adjusts each spectrum relative to a reference spectrum (usually the mean).
* Reduces non-chemical effects, preserving the relevant spectral information.

---

#### **SNV – Standard Normal Variate**

**SNV** normalizes each spectrum individually:

* Subtracts the spectrum's mean.
* Divides by the standard deviation.

This reduces scattering and scale effects, making the spectra more comparable to each other.

---

### **Filter Applied to Generated Spectra**

Mestre, after generation, a **filter is applied to the synthetic spectra** to remove physically incoherent or noisy samples.

This filter essentially calculates the Euclidean distance between the synthetic generated spectrum and all real spectra, and selects those with the smallest distance.

The goal is to ensure that only plausible spectra are added to the dataset.

---

### **First Derivative Penalty**

In addition to the traditional GAN and VAE losses, the training includes a **penalty based on the spectrum's first derivative**.

Core idea:

* Real NIR spectra tend to be **smooth**.
* Large point-to-point variations indicate noise or artifacts.

Conceptual implementation:

* The first derivative of the real and generated spectra is calculated.
* The difference between these derivatives is penalized.

Effect:

* Reduces artificial oscillations.
* Forces the generator to respect spectral continuity.

---

### **Summary of Operation**

1.  Real spectra are preprocessed (MSC / SNV).
2.  The Encoder learns the latent distribution.
3.  The Generator creates spectra conditioned on the labels.
4.  The Discriminator evaluates real vs. synthetic.
5.  Training considers:
    * Adversarial Loss (GAN)
    * Reconstruction Loss (VAE)
    * KL Divergence
    * First Derivative Penalty
6.  The generated spectra pass through a final filter before use.

### **Idea for this Repo**

1. I want to compare this approach in various datasets
2. For now i have 4 different datasets (found on google datasets)
3. Creating a solid database of results in order to write a paper about this topic