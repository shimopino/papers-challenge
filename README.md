# Paper Reading List

## experiments

Reproduction experiments for each paper were conducted on VSCode and Colab. In the top hierarchy of this repository, I have added a template configuration file for development using the Remote Container function of VSCode.

To run the experiment, please call the "Open Repository in Container" and run the file in a separate Git environment.

## TOC

- [Paper Reading List](#paper-reading-list)
  - [Generative Adversarial Networks](#generative-adversarial-networks)
    - [Image Generation](#image-generation)
    - [Image-to-Image Translation](#image-to-image-translation)
    - [Text-to-Image Translation](#text-to-image-translation)
    - [Super-Resolution](#super-resolution)
    - [Image Editting](#image-editting)
    - [Denoising](#denoising)
    - [Disentanglement](#disentanglement)
    - [Consistency Regularization](#consistency-regularization)
    - [GAN Architecute Search](#gan-arch-search)
    - [Survey](#gan-survey)
  - [Computer Vision](#computer-vision)
    - [Learning Strategy](#learn-strg)
    - [Convolution Architecture](#conv-arch)
    - [Architecure Search](#arch-search)
    - [Data Augmentation](#data-augmentation)
    - [Object Detection](#object-detection)
    - [Self-Supervised Learning](#cv-self-sup)
    - [Anomaly Detection](#anomaly-detection)
  - [Natural Language Processing](#nlp)
    - [Transformers](#transformers)
    - [Survey](#nlp-survey)
  - [Tabular](#table)
    - [Concept Shift](#con-shift)
<!-- /TOC -->

---

<a id="markdown-generative-adversarial-networks" name="generative-adversarial-networks"></a>
## Generative Adversarial Networks

<a id="markdown-image-generation" name="image-generation"></a>
### Image Generation

- [[arXiv:1809.11096] Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/106)
- [[arXiv:1905.01164] SinGAN: Learning a Generative Model from a Single Natural Image](https://arxiv.org/abs/1905.01164)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/1)
- [[arXiv:1912.11035] CNN-generated images are surprisingly easy to spot... for now](https://arxiv.org/abs/1912.11035)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/21)
- [[arXiv:2002.10964] Freeze the Discriminator: a Simple Baseline for Fine-Tuning GANs](https://arxiv.org/abs/2002.10964)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/16)
- [[arXiv:2002.12655] A U-Net Based Discriminator for Generative Adversarial Networks](https://arxiv.org/abs/2002.12655)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/17)
- [[arXiv:2004.02088] Feature Quantization Improves GAN Training](https://arxiv.org/abs/2004.02088)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/62)
- [[arXiv:2003.02567] GANwriting: Content-Conditioned Generation of Styled Handwritten Word Images](https://arxiv.org/abs/2003.02567)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/24)
- [[arXiv:2004.03355] Inclusive GAN: Improving Data and Minority Coverage in Generative Models](https://arxiv.org/abs/2004.03355)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/65)
- [[arXiv:2004.05472] Autoencoding Generative Adversarial Networks](https://arxiv.org/abs/2004.05472)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/81)
    

<a id="markdown-image-to-image-translation" name="image-to-image-translation"></a>
### Image-to-Image Translation

- [[arXiv:1611.07004] Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/29)
- [[arXiv:1711.09020] StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/101)
- [[arXiv:1711.11585] High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://arxiv.org/abs/1711.11585)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/30)
- [[arXiv:1812.10889] InstaGAN: Instance-aware Image-to-Image Translation](https://arxiv.org/abs/1812.10889)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/100)
- [[arXiv:1907.04312] Positional Normalization](https://arxiv.org/abs/1907.04312)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/52)
- [[arXiv:2002.05638] GANILLA: Generative Adversarial Networks for Image to Illustration Translation](https://arxiv.org/abs/2002.05638)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/9)
- [[arXiv:2003.00187] Augmented Cyclic Consistency Regularization for Unpaired Image-to-Image Translation](https://arxiv.org/abs/2003.00187)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/19)
- [[arXiv:2003.02683] Image Generation from Freehand Scene Sketches](https://arxiv.org/abs/2003.02683)
- [[arXiv:2003.00273] Reusing Discriminators for Encoding Towards Unsupervised Image-to-Image Translation](https://arxiv.org/abs/2003.00273)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/26)
- [[arXiv:2003.04858] Unpaired Image-to-Image Translation using Adversarial Consistency Loss](https://arxiv.org/abs/2003.04858)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/45)
- [[arXiv:2003.07101] Synthesizing human-like sketches from natural images using a conditional convolutional decoder](https://arxiv.org/abs/2003.07101)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/44)


<a id="markdown-text-to-image-translation" name="text-to-image-translation"></a>
### Text-to-Image Translation

- [[arXiv:1612.03242] StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/abs/1612.03242)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/7)
- [[arXiv:1802.09178] Photographic Text-to-Image Synthesis with a Hierarchically-nested Adversarial Network](https://arxiv.org/abs/1802.09178)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/33)
- [[arXiv:1904.01480] Semantics Disentangling for Text-to-Image Generation](https://arxiv.org/abs/1904.01480)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/34)
- [[arXiv:2003.12137] Cycle Text-To-Image GAN with BERT](https://arxiv.org/abs/2003.12137)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/58)
- [[arXiv:2004.11437] Efficient Neural Architecture for Text-to-Image Synthesis](https://arxiv.org/abs/2004.11437)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/97)

<a id="markdown-super-resolution" name="super-resolution"></a>
### Super-Resolution

- [[arXiv:2003.02365] Creating High Resolution Images with a Latent Adversarial Generator](https://arxiv.org/abs/2003.02365)
     - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/23)
- [[arXiv:2004.00448] Rethinking Data Augmentation for Image Super-resolution: A Comprehensive Analysis and a New Strategy](https://arxiv.org/abs/2004.00448)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/64)

<a id="markdown-image-editting" name="image-editting"></a>
### Image Editting

- [[arXiv:2004.00049] In-Domain GAN Inversion for Real Image Editing](https://arxiv.org/abs/2004.00049)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/60)

<a id="markdown-denoising" name="denoising"></a>
### Denoising

- [[arXiv:2003.07849] Blur, Noise, and Compression Robust Generative Adversarial Networks](https://arxiv.org/abs/2003.07849)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/47)

<a id="markdown-disentanglement" name="disentanglement"></a>
### Disentanglement

- [[arXiv:2002.03754] Unsupervised Discovery of Interpretable Directions in the GAN Latent Space](https://arxiv.org/abs/2002.03754)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/8)

<a id="markdown-consistency-regularization" name="consistency-regularization"></a>
### Consistency Regularization

- [[arXiv:1904.12848] Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/20)
- [[arXiv:2002.04724] Improved Consistency Regularization for GAN](https://arxiv.org/abs/2002.04724)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/11)

<a id="markdown-gan-arch-search" name="gan-arch-search"></a>
### GAN Architecture Search
- [[arXiv:2003.08936] GAN Compression: Efficient Architectures for Interactive Conditional GANs](https://arxiv.org/abs/2003.08936)
  - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/48)

<a id="markdown-gan-survey" name="gan-survey"></a>
### Survey
- [[arXiv:1906.01529] Generative Adversarial Networks in Computer Vision: A Survey and Taxonomy](https://arxiv.org/abs/1906.01529)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/15)

---

<a id="markdown-computer-vision" name="computer-vision"></a>
## Computer Vision

<a id="markdown-learn-strg" name="learn-strg"></a>
### Learning Strategy
- [[arXiv:1911.09665] Adversarial Examples Improve Image Recognition](https://arxiv.org/abs/1911.09665)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/3)
- [[arXiv:2004.11362] Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/94)
- [[arXiv:2003.00152] Training BatchNorm and Only BatchNorm: On the Expressive Power of Random Features in CNNs](https://arxiv.org/abs/2003.00152)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/102)

<a id="markdown-conv-arch" name="conv-arch"></a>
### Convolution Architecture

- [[arXiv:1811.12231] ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness](https://arxiv.org/abs/1811.12231)
- [[arXiv:1906.05909] Stand-Alone Self-Attention in Vision Models](https://arxiv.org/abs/1906.05909)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/13)
- [[arXiv:1911.08432] Defective Convolutional Layers Learn Robust CNNs](https://arxiv.org/abs/1911.08432)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/2)
- [[arXiv:2003.01367] Curriculum By Texture](https://arxiv.org/abs/2003.01367)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/36)
- [[arXiv:2004.13587] Do We Need Fully Connected Output Layers in Convolutional Networks?](https://arxiv.org/abs/2004.13587)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/107)

<a id="markdown-arch-search" name="arch-search"></a>
### Architecture Search
- [[arXiv:1908.09791] Once-for-All: Train One Network and Specialize it for Efficient Deployment on Diverse Hardware Platforms](https://arxiv.org/abs/1908.09791)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/49)

<a id="markdown-data-augmentation" name="data-augmentation"></a>
### Data Augmentation
- [[arXiv:1905.04899] CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/abs/1905.04899)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/18)
- [[arXiv:1909.09148] Data Augmentation Revisited: Rethinking the Distribution Gap between Clean and Augmented Data](https://arxiv.org/abs/1909.09148)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/4)
- [[arXiv:2002.11102] On Feature Normalization and Data Augmentation](https://arxiv.org/abs/2002.11102)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/51)

<a id="markdown-object-detection" name="object-detection"></a>
### Object Detection
- [[arXiv:2003.05176] Equalization Loss for Long-Tailed Object Recognition](https://arxiv.org/abs/2003.05176)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/41)
- [[arXiv:2004.08955] ResNeSt: Split-Attention Networks](https://arxiv.org/abs/2004.08955)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/92)

<a id="markdown-cv-self-sup" name="cv-self-sup"></a>
### Self-Supervised Learning
- [[arXiv:2002.05709] A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/38)
- [[arXiv:2003.04297] Improved Baselines with Momentum Contrastive Learning](https://arxiv.org/abs/2003.04297)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/46)

<a id="markdown-anomaly-detection" name="anomaly-detection"></a>
### Anomaly Detection
- [[arXiv:1903.08550] OCGAN: One-class Novelty Detection Using GANs with Constrained Latent Representations](https://arxiv.org/abs/1903.08550)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/6)

<a id="markdown-nlp" name="nlp"></a>
## Natural Language Processing

<a id="markdown-transformers" name="transformers"></a>
### Transformers
- [[arXiv:2003.07845] Rethinking Batch Normalization in Transformers](https://arxiv.org/abs/2003.07845)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/50)
- [[arXiv:2002.11794] Train Large, Then Compress: Rethinking Model Size for Efficient Training and Inference of Transformers](https://arxiv.org/abs/2002.11794)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/56)
- [[arXiv:2004.02984] MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://arxiv.org/abs/2004.02984)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/80)
- [[arXiv:2004.11886] Lite Transformer with Long-Short Range Attention](https://arxiv.org/abs/2004.11886)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/96)
- [[arXiv:2005.00743] Synthesizer: Rethinking Self-Attention in Transformer Models](https://arxiv.org/abs/2005.00743)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/109)
    
<a id="markdown-nlp-survey" name="nlp-survey"></a>
### Survey
- [[arXiv:2004.08900] The Cost of Training NLP Models: A Concise Overview](https://arxiv.org/abs/2004.08900)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/89)


<a id="markdown-table" name="table"></a>
## Tabular

<a id="markdown-con-shift" name="con-shift"></a>
### Concept Shift

- [[arXiv:2004.03045] Adversarial Validation Approach to Concept Drift Problem in Automated Machine Learning Systems](https://arxiv.org/abs/2004.03045)
    - [issue](https://github.com/KeisukeShimokawa/papers-challenge/issues/68)
