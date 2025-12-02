## Automated detection of racket–ball impact timing in tennis strokes using deep learning models and a markerless motion capture system.
Tennis stroke impact timing prediction


#### Abstract
Defining time events is a fundamental step in biomechanics research, as it enables precise analysis of dynamic motion and its underlying mechanisms. This study aimed to develop and evaluate deep learning models for automated detection of the impact event-the moment of racket-ball contact-during tennis strokes. For this purpose, Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks were implemented using kinematic data collected from a markerless motion capture system. Kinematic data were obtained from 16 elite tennis players performing standard stroke techniques. Joint center coordinates extracted from the markerless system served as time-series features for model input. The models successfully detected the impact event within ±1 ms of ground-truth annotation. Among various joint combinations, the use of ankle (ANK), elbow (ELB), knee (KNEE), and wrist (WRI) provided the highest performance, with overall classification accuracy exceeding 95%. For this optimal combination, the LSTM model achieved a precision of 95.77%, an F1-score of 97.09%, and an AUC of 98.06%, while the GRU model achieved a precision of 96.23%, an F1-score of 97.71%, and an AUC of 99.20%. These findings demonstrate that automated detection of sport-specific biomechanical events can effectively replace labor-intensive manual annotation. Moreover, markerless motion capture systems enable large-scale and ecologically valid data collection, offering a viable alternative to laboratory-based methods. The proposed approach provides a methodological basis for real-time monitoring and practical feedback in tennis performance analysis and training.

<img width="1242" height="653" alt="Figure_3" src="https://github.com/user-attachments/assets/e5b32b13-d549-46f9-8683-1ec2baae474a" />



### 2. Environment

All analyses were performed in a Windows 11 PC environment using the PyTorch framework.  
Model training and testing were conducted on a workstation equipped with:

- GPU: NVIDIA RTX 3090 (24 GB)
- CPU: 24-thread processor
- RAM: 128 GB
- Python: 3.11.7

To install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Dataset

The dataset consists of time-series kinematic features extracted from a markerless motion capture system.

Input features (per frame)
3D joint centers of:

Left ankle, knee, elbow

Right ankle, knee, elbow
→ 6 joints × 3 coordinates = 18 features per frame

Target label

binary column

1: impact frame (racket–ball contact)

0: non-impact frame



### 4. Project Structure

tennis-DL_tennis/
├─ README.md
├─ requirements.txt
├─ .gitignore
│
├─ src/
│  ├─ __init__.py
│  ├─ datasets.py     # data loading, preprocessing, sequence generation
│  ├─ models.py       # LSTM and GRU model definitions
│  └─ train.py        # main script: training, evaluation, visualization
│
├─ data/
│  └─ README.md       # description of sliced_data.csv (no raw data uploaded)
│
└─ results/           # (optional) logs, figures, checkpoints (git-ignored)



### 5. Training & Evaluation
python -m src.train

Load data/sliced_data.csv

Normalize features with StandardScaler

Construct MANY-to-ONE sequences of length 50

Balance the classes via downsampling

Split the data into training and test sets using StratifiedShuffleSplit

Train both LSTM and GRU models

Evaluate:

Accuracy

Confusion matrices

ROC curves and AUC

Precision, recall, and F1-scores

Visualize the results using matplotlib and seaborn


### 6. Reproducing Paper Results

The default hyperparameters in src/train.py, src/models.py, and src/datasets.py correspond to the settings used to produce the main results in the manuscript (e.g., hidden size, number of layers, dropout rate, sequence length, batch size).

If you change these settings, please document them for reproducibility.

