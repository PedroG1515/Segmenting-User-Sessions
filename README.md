# Segmenting User Sessions in Search Engine Query Logs Leveraging Word Embeddings

Reproducibility package for the TPDL paper "Segmenting User Sessions in Search Engine Query Logs Leveraging Word Embeddings"

### Setup

```
git clone https://github.com/PedroG1515/Segmenting-User-Sessions.git
cd Segmenting-User-Sessions
pip install --upgrade virtualenv
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

### Models

Download the following models into your `Segmenting-User-Sessions` directory:

```
https://figshare.com/articles/wiki_normal_bin_vectors_ngrams_npy/8036168
https://figshare.com/articles/wiki_with_vectors_normalized_bin_vectors_ngrams_npy/8036102
```

### Run
```
python Proposed_Method_with_WMD.py
```

### Authors

Pedro Gomes
Luis Cruz
Bruno Martins
