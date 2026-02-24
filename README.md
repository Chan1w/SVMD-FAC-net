# SVMD-FAC-Net

Official implementation of **SVMD-FAC-Net**, a hybrid decomposition–transformer framework for lithium-ion battery State-of-Health (SOH) prediction.

---

## Abstract

Accurate prediction of battery State-of-Health (SOH) is critical for reliability and safety in electric vehicle applications. 
However, battery degradation signals are typically nonlinear, nonstationary, and exhibit multi-scale temporal characteristics.

To address these challenges, we propose **SVMD-FAC-Net**, a hybrid modeling framework that:

- Utilizes **Successive Variational Mode Decomposition (SVMD)** to extract intrinsic multi-scale components,
- Integrates a modified **Autoformer architecture** for long-term temporal dependency modeling,
- Introduces a **Frequency-Aware Correction (FAC) module** to enhance attention robustness in decomposed feature space.

Extensive experiments demonstrate that the proposed framework achieves superior prediction accuracy and stability compared with conventional recurrent and transformer-based baselines.

---

## Methodology

The overall pipeline of SVMD-FAC-Net is illustrated as follows:


### Key Components

1. **SVMD Module**
   - Decomposes raw degradation signals into intrinsic mode components
   - Captures multi-scale degradation patterns

2. **FAC Module**
   - Adjusts attention distribution in frequency domain
   - Improves robustness under nonstationary dynamics

3. **Modified Autoformer**
   - Long-sequence modeling
   - Series decomposition mechanism
   - Efficient self-attention structure

---

## Repository Structure


---

## Installation

```bash
git clone https://github.com/yourname/SVMD-FAC-Net.git
cd SVMD-FAC-Net
pip install -r requirements.txt
