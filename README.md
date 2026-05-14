## CVAE-GAN for NIR Spectra Generation

This repository implements a **CVAE-GAN** (Conditional Variational Autoencoder + Generative Adversarial Network) used in my final graduation project.

The model was designed to work with one-dimensional spectral data (vectors) and conditional labels, making it suitable for **data augmentation** tasks in regression problems.

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

# Notes:

