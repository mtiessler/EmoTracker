# **RoVERTaD: VAD Inference Over Time with Diachronic Sense Modeling**


This project is **still in development**. The following components are functional or under active construction:

- Dataset generation based on Hu et al.’s temporal senses and NRC-VAD is **in progress**
- RoBERTa training and evaluation pipeline is **in progress**
- Frontend visualization client is **in progress**

## **Table of Contents**

1. [General Description](#general-description)  
2. [Detailed Process](#detailed-process)  
   2.1. [Data Acquisition from Hu et al.](#1-data-acquisition-from-hu-et-al)  
   2.2. [VAD Analysis for Senses](#2-vad-analysis-for-senses)  
   2.3. [Resulting Dataset Generation](#3-resulting-dataset-generation)  
   2.4. [3D Plot Visualization](#4-3d-plot-visualization)  
   2.5. [RoBERTa Model Training for Temporal VAD Inference](#5-roberta-model-training-for-temporal-vad-inference)  
3. [Threats to Validity](#threats-to-validity)  
4. [Project Structure](#project-structure)  
5. [Installation](#installation)  
6. [Usage](#usage)  

---

## **General Description**

This repository contains the code and resources necessary to perform the inference of **Valence**, **Arousal**, and **Dominance** (VAD) values over time using a RoBERTa model trained on a dataset derived from the diachronic sense modeling of Hu et al.

The process involves acquiring historical sense data, assigning VAD values to these senses, generating a dataset of time-weighted VAD values for words, and visualizing these emotional trajectories in a 3D space.

Finally, this dataset is used to fine-tune a RoBERTa model for temporal VAD inference.

---

## **Detailed Process**

### **1. Data Acquisition from Hu et al.**

- The diachronic sense modeling data provided by Hu et al. is utilized.
- This data is available on their GitHub repository: [[iris2hu/diachronic-sense-modeling](https://github.com/iris2hu/diachronic-sense-modeling)].
- The main file, `prob_fitting_10.data`, contains information about the proportions of different senses of words over time (1830–2010, in 10-year intervals).
- The file `poly_vocab.txt` lists the 3,220 polysemous words analyzed.
- The dataset is based on the COHA (Corpus of Historical American English).

---

### **2. VAD Analysis for Senses**

- A VAD lexicon is used to assign Valence (positivity/negativity), Arousal (calmness/excitement), and Dominance (sense of control) values to the definitions of word senses.
- Existing VAD lexicons include:
  - **ANEW** and its extension by Warriner et al. (2013), with ~14,000 entries. Available via [[JULIELab/XANEW](https://github.com/JULIELab/XANEW)].
  - (**NRC-VAD**)[https://saifmohammad.com/WebPages/nrc-vad.html], updated in 2025, with ~50,000 English words.
  - **EmoBank**, which provides sentence-level VAD annotations—useful for contextual analysis but not directly for sense definitions.

For each sense definition, the keywords are matched against a VAD lexicon. If multiple matches are found, the average VAD values are used to approximate the emotion of that sense.  
In this first iteration we have used **NRC-VAD** lexicon.

---

### **3. Resulting Dataset Generation**

- Each word sense is assigned a fixed VAD value.
- Using Hu et al.'s time-based sense proportions, we compute the overall VAD value of each word at each time step.
- The assumption is that while the emotional quality of each sense remains constant, the prominence of each sense changes over time.

The final word-level VAD value at time **t** is computed as a weighted average:

\[
VAD(w, t) = \sum_{i=1}^{n} P(s_i, t) \times VAD(s_i)
\]

Where:
- \( VAD(w, t) \) is the VAD vector for word \( w \) at time \( t \)
- \( P(s_i, t) \) is the proportion of sense \( s_i \) at time \( t \)
- \( VAD(s_i) \) is the fixed VAD vector for sense \( s_i \)

This results in a diachronic dataset of words with corresponding VAD trajectories.

---

### **4. 3D Plot Visualization**

Once temporal VAD values are computed, they can be visualized in a 3D space with Valence, Arousal, and Dominance as axes.

- Each point on a word's trajectory represents its VAD at a given time.
- This enables intuitive observation of emotional drift over time.

*Note: Visualization implementation is in progress — options include static plots via matplotlib or interactive React-based clients.*

---

### **5. RoBERTa Model Training for Temporal VAD Inference**

- A pre-trained RoBERTa model is fine-tuned using the generated diachronic VAD dataset.
- Inspired by Park et al.'s approach to emotion detection via RoBERTa (using Earth Mover’s Distance for training), we aim to adapt their method for VAD regression over time.
- The training data would ideally consist of historical contexts (e.g., COHA sentences) paired with the time-specific VAD values derived from our pipeline.

Reference: [[SungjoonPark/EmotionDetection](https://github.com/SungjoonPark/EmotionDetection)]

The resulting model should accept a word (or short historical context) and a time period, returning a predicted VAD vector for that era.

---

## **Threats to Validity**

A major limitation of this approach is the lack of a standardized, gold-standard dataset of VAD values across historical periods. The only notable work in this direction is Buechel et al.’s dataset for **German**, spanning **1690 to 1899**.

As a result, this project makes the **strong assumption** that the emotional quality of a sense (its VAD value) remains constant over time — which is likely not the case. Emotional connotations of words evolve, and this assumption introduces a source of noise into our generated dataset.

To train a temporally accurate RoBERTa model, **expert annotation** of VAD values should be performed for the senses in Hu et al.’s dataset across **20-year intervals** covering 1830–2010. This would establish a reliable ground truth and significantly improve model performance and trustworthiness.

---

## **Project Structure**

```plaintext
RoVertAD/
├── data/                              # External data sources
│   ├── Diachronic_Sense_Modeling/    # Hu et al.'s sense probabilities
│   │   └── prob_fitting_10.data
│   └── VAD_Lexicons/
│       └── NRC-VAD-Lexicon-v2.1/
│           ├── NRC-VAD-Lexicon-v2.1.txt
│           └── README.txt
│
├── env/                               # Virtual environment (optional)
│
├── src/                               # Core source code
│   ├── model/
│   │   ├── config.py                  # Config and hyperparameters
│   │   ├── data_loader.py            # Loads time-VAD dataset
│   │   ├── model.py                  # RoBERTa model wrapper
│   │   ├── trainer.py                # Training loop and loss functions
│   │   ├── utils.py                  # Helper functions
│   │   └── main.py                   # Entrypoint to training/inference
│   └── scripts/
│       └── format_converter.py       # Convert Hu et al.’s format to usable data
│
├── visualization_client/             # Frontend visualization (React app)
│   ├── package.json                  # React dependencies and scripts
│   └── ...                           # React components and pages
│
├── .env
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt                  # Python dependencies
```

---

## **Installation**

### 1. Clone the repository

```bash
git clone https://github.com/yourname/RoVertAD.git
cd RoVertAD
```

### 2. Create a virtual environment

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install frontend dependencies

```bash
cd visualization_client
npm install
```

---

