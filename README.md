# Fake Political News Detection Using NLP

**IE7500 Applied NLP for Engineers - Course Project**  
**Student:** Cosmos Ameyaw Kwakye  
**Semester:** Fall 2025

## Project Overview

This project implements a fake political news detection system using Natural Language Processing techniques on the LIAR dataset, which contains 12,791 political statements labeled for veracity.

## Current Status (Week 9 - Mid-Project)

- ✅ **Phase 1 Complete:** Data preprocessing pipeline
- ✅ **Phase 2 Complete:** Baseline models (TF-IDF + Naive Bayes/Logistic Regression)
- ✅ **Phase 3 Complete:** Word embeddings (GloVe) + Multi-Layer Perceptron
- ⏳ **Phase 4 Planned:** BERT fine-tuning (Weeks 10-12)

## Performance Results

| Model | Test Accuracy | F1-Score | Status |
|-------|--------------|----------|--------|
| Naive Bayes (TF-IDF) | 61.01% | 0.6916 | ✅ Best Baseline |
| Logistic Regression | 61.45% | 0.6275 | ✅ Complete |
| MLP + GloVe Embeddings | 60.77% | 0.6343 | ✅ Complete |
| BERT (planned) | Target: 68%+ | Target: 0.75+ | ⏳ Next Phase |

## Dataset

**LIAR Dataset** (Wang, 2017)
- **Source:** PolitiFact fact-checking website
- **Size:** 12,791 political statements
- **Split:** 
  - Training: 10,240 (80%)
  - Validation: 1,284 (10%)
  - Test: 1,267 (10%)
- **Labels:** Binary classification (Fake vs. Real)
- **Download:** Available on Kaggle

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Google Colab (recommended) OR Jupyter Notebook
- 8GB RAM minimum

### Installation

1. **Clone this repository:**
```bash
   git clone https://github.com/KwakyeCA/fake-political-news-detection-nlp.git
   cd (https://github.com/KwakyeCA/fake-political-news-detection-nlp)
```

2. **Install required packages:**
```bash
   pip install pandas numpy matplotlib seaborn
   pip install scikit-learn
   pip install tensorflow
   pip install nltk
```

3. **Download NLTK data:**
```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
```

4. **Download GloVe embeddings** (for Phase 3):
   - Download: [GloVe 6B](https://nlp.stanford.edu/data/glove.6B.zip)
   - Extract `glove.6B.300d.txt`
   - Or run the download cell in the notebook (automated)

### Data Setup

1. Download LIAR dataset files:
   - `train.tsv`
   - `valid.tsv`
   - `test.tsv`

2. Place them in the same directory as the notebook, OR
3. Upload them when prompted in Colab

## Usage

### Option 1: Google Colab (Recommended)

1. Open the notebook in Colab:
   - Upload `Fake_News_Detection_Complete.ipynb` to Colab
   - Or use: File → Upload notebook

2. Upload dataset files when prompted

3. Run all cells sequentially:
   - Runtime → Run all

### Option 2: Local Jupyter

1. Start Jupyter:
```bash
   jupyter notebook
```

2. Open `Fake_News_Detection_Complete.ipynb`

3. Run cells in order

## Project Structure
```
fake-political-news-detection/
├── Fake_Political_News_Detection_Complete.ipynb  # Main notebook (all phases)
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
└── data/                              # Dataset folder (not included)
    ├── train.tsv
    ├── valid.tsv
    └── test.tsv
```

## Methodology

### Phase 1: Data Preprocessing
- Text cleaning (lowercase, punctuation removal)
- Tokenization using NLTK
- Stopword removal and lemmatization
- Binary label creation (Fake vs. Real)

### Phase 2: Baseline Models
- **TF-IDF Vectorization:** 5,000 features, unigram + bigram
- **Models:**
  - Logistic Regression with hyperparameter tuning
  - Multinomial Naive Bayes (selected as best baseline)

### Phase 3: Neural Network with Embeddings
- **Embeddings:** GloVe 6B.300d (400,000 word vectors)
- **Architecture:** 
  - Embedding layer (300d, trainable)
  - Global average pooling
  - Dense layers: 256 → 128 → 64
  - Batch normalization + dropout
- **Training:** Early stopping, learning rate reduction

### Phase 4: BERT (Upcoming)
- Fine-tuning pre-trained BERT model
- Target: 68%+ accuracy
- Weeks 10-12

## Key Findings

**Research Insight:** Static word embeddings (GloVe) performed slightly worse than TF-IDF baseline, revealing that:
1. Small dataset size (10K samples) favors simpler models
2. Short text length (~18 words) limits embedding effectiveness
3. Keyword-based features outperform semantic features for this task
4. Contextual embeddings (BERT) needed for improvement

This finding aligns with published literature on LIAR dataset and motivates Phase 4.

## Results & Visualizations

The notebook includes:
- 📊 Data exploration and statistics
- 📈 Training history plots
- 🎯 Confusion matrices
- 📉 ROC curves
- 🔍 Feature importance analysis
- 📋 Model comparison tables

## Technologies Used

- **Languages:** Python 3.8+
- **Libraries:**
  - Data: pandas, numpy
  - NLP: NLTK, scikit-learn
  - Deep Learning: TensorFlow/Keras
  - Embeddings: GloVe
  - Visualization: matplotlib, seaborn

## References

Wang, W. Y. (2017). "Liar, liar pants on fire": A new benchmark dataset for fake news detection. *Proceedings of ACL*, 422–426.

Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. *Proceedings of EMNLP*, 1532-1543.

## Timeline

- ✅ **Weeks 3-4:** Proposal & data collection
- ✅ **Weeks 5-6:** Baseline models
- ✅ **Weeks 7-9:** Embeddings + MLP
- ⏳ **Weeks 10-12:** BERT implementation
- ⏳ **Weeks 13-14:** Comprehensive analysis
- ⏳ **Weeks 15-16:** Final report & presentation

## Contact

**Cosmos Ameyaw Kwakye**  
Northeastern University  
Email: [kwakye.c@northeastern.edu]

## License

This project is for educational purposes as part of IE7500 coursework.

---

**Last Updated:** October 27, 2025  
**Status:** Phase 3 Complete (60% of project)
