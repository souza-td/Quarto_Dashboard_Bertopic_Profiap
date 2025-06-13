# Imports e caminhos
from pathlib import Path
import json, pandas as pd
from octis.dataset.dataset import Dataset
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

ROOT = Path('./')
DATA_DIR   = ROOT / 'data'
MODEL_DIR  = ROOT / 'models'

# Dataset Octis
octis_ds = Dataset()
octis_ds.load_custom_dataset_from_folder(str(DATA_DIR / 'octis'))
texts = octis_ds.get_corpus()
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(t) for t in texts]

# Avaliação
catalog = pd.read_csv(MODEL_DIR / 'runs_catalog.csv')
scores = []

for _, row in catalog.iterrows():
    topic_words = json.load(open(row['topics_path'], encoding='utf-8'))

    score_dict = {
        'n_neighbors' : row['n_neighbors'],
        'cluster_size': row['cluster_size'],
        'n_topics':     row['n_topics'],
        'c_v':    CoherenceModel(topics=topic_words, texts=texts,
                                 dictionary=dictionary,
                                 coherence='c_v').get_coherence(),
        'c_npmi': CoherenceModel(topics=topic_words, texts=texts,
                                 dictionary=dictionary,
                                 coherence='c_npmi').get_coherence(),
        'u_mass': CoherenceModel(topics=topic_words, corpus=corpus,
                                 dictionary=dictionary,
                                 coherence='u_mass').get_coherence(),
    }

    uniq = len({w for t in topic_words for w in t})
    score_dict['diversity'] = uniq / (len(topic_words) * len(topic_words[0]))
    scores.append(score_dict)

metrics_df = pd.DataFrame(scores).sort_values('cluster_size')
metrics_df.to_csv(MODEL_DIR / 'metrics_by_cluster_size.csv', index=False)
metrics_df

