# Segmenting User Sessions in Search Engine Query Logs Leveraging Word Embeddings

Reproducibility package for the TPDL paper _"Segmenting User Sessions in Search Engine Query Logs Leveraging Word Embeddings"_.

### Abstract

Segmenting user sessions in search engine query logs is important to perceive information needs and assess how they are satisfied, to enhance the quality of search engine rankings, and to better direct content to certain users. Most previous methods use human judgments to inform supervised learning algorithms, and/or use global thresholds on temporal proximity and on simple lexical similarity metrics. This paper proposes a novel unsupervised method that improves the current state-of-art, leveraging additional heuristics and similarity metrics derived from word embeddings. We specifically extend a previous approach based on combining temporal and lexical similarity measurements, integrating semantic similarity components that use pre-trained FastText embeddings. The paper reports on experiments with an AOL query dataset used in previous studies, containing a total of 10,235 queries, with 4,253 sessions, 2.4 queries per session, and 215 unique users. The results attest to the effectiveness of the proposed method, which outperforms a large set of baselines, also corresponding to unsupervised techniques.

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

- [Pedro Gomes](mailto:pedro.almeida.gomes@tecnico.ulisboa.pt)
- Luis Cruz
- Bruno Martins
