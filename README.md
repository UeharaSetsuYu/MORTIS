# MORTIS: Reciprocal Cyclic Calibration for Incomplete and Noisy Multi-View Clustering

 
MORTIS is an imputation-free multi-view clustering framework designed for
incomplete, noisy, and compound incomplete-and-noisy data. It stabilizes
prototype estimation by preserving reliable neighborhood structures and
calibrates cross-view prototypes through reciprocal cyclic transitions.


``` 
python Run_model.py --dataset Caltech5V --missing_rate 0.5 --data_model incomplete
```



```text
MORTIS/
├── Run_model.py               # Main experimental entry point
├── train.py                   # Training and evaluation pipeline
├── model.py                   # Network architecture
├── data/                      # Multi-view datasets
├── units/
│   ├── config.py              # Dataset-specific configurations
│   ├── IMVC_DATA.py           # Incomplete data construction
│   ├── NMVC_DATA.py           # Noisy data construction
│   ├── CMVC_DATA.py           # Compound degradation construction
│   ├── loss.py                # Training objectives
│   ├── clustering.py          # Clustering utilities
│   └── evaluate.py            # Evaluation metrics
├── Metric/                    # Prototype and clustering analyses
├── logs/                      # Training logs
└── README.md
```
Requirements
The code is implemented in Python with PyTorch. Install PyTorch according
to your CUDA version and then install the remaining dependencies:
```
pip install numpy scipy scikit-learn pandas h5py
pip install matplotlib seaborn tqdm munkres scanpy openpyxl
```

