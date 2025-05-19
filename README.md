# GIMMEA

# Dataset Preparation Guide

## 📂 Bilingual Datasets (DBP15K)

### Source
The multi-modal version of DBP15K dataset comes from the [EVA repository](https://github.com/your-repo-link).

### Setup Instructions:
1. **Image Features**:
   - Download the `pktS` folder containing DBP15K image features according to EVA repository's guidance
   - Place the downloaded folder in the `data` directory of this repository

2. **Word Embeddings**:
   ```bash
   # Download and install GloVe embeddings
   wget https://nlp.stanford.edu/data/glove.6B.zip
   unzip glove.6B.zip -d data/embedding/
