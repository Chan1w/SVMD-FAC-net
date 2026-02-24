🚀 SVMD-FAC-net
A Lightweight Multiscale Signal Learning Framework for Battery Degradation Trajectory Prediction

📖 Introduction

This repository provides the official implementation of:

A Lightweight Multiscale Signal Learning Framework for Predicting Battery Degradation Trajectory
IEEE Sensors Journal, Vol. 25, No. 24, Dec. 2025

The paper proposes a novel degradation trajectory prediction framework integrating:

🔹 Successive Variational Mode Decomposition (SVMD)

🔹 Period-Sensitive Auto-Correlation Module (ACM)

🔹 Lightweight MLP-based RES Prediction Network

🔹 Bayesian Hyperparameter Optimization

The method achieves:

📉 MAE < 0.59%

📉 RMSE < 0.96%

⚡ Low training & inference time

💡 Lightweight & deployment-friendly architecture

🧠 Framework Overview

The proposed framework consists of four main steps:

Raw Capacity Sequence
        │
        ▼
   SVMD Decomposition
   ├── IMFs (high-frequency)
   └── RES  (low-frequency)
        │
        ▼
 Parallel Lightweight Networks
   ├── ACM-based IMF Predictor
   └── MLP-based RES Predictor
        │
        ▼
  Trajectory Reconstruction
Key Design Ideas

✔ Multiscale signal decoupling
✔ Periodicity-sensitive autocorrelation modeling
✔ Simplified QK-based attention (no V matrix)
✔ Lightweight spatiotemporal MLP design
✔ Bayesian optimization for hyperparameters

📊 Supported Datasets

The experiments are conducted on:

CALCE Dataset

NASA Battery Dataset

You may download the datasets from:

NASA: http://ti.arc.nasa.gov/project/prognostic-data-repository

CALCE: University of Maryland CALCE battery dataset

Please place datasets under:

data/
 ├── CALCE/
 └── NASA/
⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/yourname/SVMD-FAC-net.git
cd SVMD-FAC-net
2️⃣ Install dependencies
pip install -r requirements.txt
🏃 Training
python train.py --dataset CALCE --battery CS2_35

Arguments:

Argument	Description
--dataset	CALCE / NASA
--battery	Battery ID
--window_size	Sliding window length
--epochs	Training epochs
🧪 Testing
python test.py --dataset CALCE --battery CS2_35

Evaluation Metrics:

MAE

RMSE

AE

RPE

📈 Performance Summary
Dataset	Avg MAE	Avg RMSE
CALCE	0.58%	0.96%
NASA	1.10%	1.59%

Compared to CNN / LSTM / TCN / Transformer:

↓ 70–80% MAE reduction

↓ 60–77% RMSE reduction

Lower inference time

🧩 Project Structure
SVMD-FAC-net/
│
├── models/
│   ├── svmd.py
│   ├── imf_network.py
│   ├── res_network.py
│   └── acm_module.py
│
├── data/
├── train.py
├── test.py
├── utils.py
├── requirements.txt
└── README.md
📌 Citation

If you find this work useful, please cite:

@article{shen2025svmd,
  title={A Lightweight Multiscale Signal Learning Framework for Predicting Battery Degradation Trajectory},
  author={Shen, Quanyong and Li, Jian and Nie, Jiahao and Bao, Zhengyi and Wang, Chenhan},
  journal={IEEE Sensors Journal},
  volume={25},
  number={24},
  pages={44801--44812},
  year={2025},
  publisher={IEEE}
}
👩‍🔬 Authors

Quanyong Shen

Jian Li

Jiahao Nie

Zhengyi Bao (Corresponding Author)

Chenhan Wang

📧 Contact: baozy@hdu.edu.cn

🔮 Future Work

Deployment in real-world EV battery systems

Edge-device optimization

Real-time BMS integration

Extension to general time-series regression tasks

📜 License

This project is released under the MIT License.

⭐ If this repository helps your research, please consider giving it a star!
