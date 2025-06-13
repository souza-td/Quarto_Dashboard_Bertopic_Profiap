# Imports e caminhos
from pathlib import Path
import json, pandas as pd, random, numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
import spacy, gensim
from spacy.lang.pt.stop_words import STOP_WORDS as STOP_PT
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

ROOT = Path('./')
DATA_DIR   = ROOT / 'data'
MODEL_DIR  = ROOT / 'models'/ 'expr_granite'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED)

# Carregar dados e pré-processar
CSV_PATH = DATA_DIR / 'raw' / 'dissertacoes_profiap_14_23.csv'  # ajuste se precisar
df = pd.read_csv(CSV_PATH).drop_duplicates("DS_RESUMO")
docs = df["DS_RESUMO"].fillna("").tolist()
nlp  = spacy.load('pt_core_news_sm', disable=['ner', 'parser'])
def preprocess(doc):
    return ' '.join([t.lemma_.lower() for t in nlp(doc) if t.is_alpha and not t.is_stop])
docs_pp = [preprocess(d) for d in docs]
# emb_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
emb_model = SentenceTransformer(
    "ibm-granite/granite-embedding-278m-multilingual",
    trust_remote_code=True          # requerido porque o código customizado está no repo
)
embeddings = emb_model.encode(docs_pp, show_progress_bar=True)

EXTRA_SW = {
    "universidade federal","universidade","federal","pesquisa","análise","estudo",
    "objetivo","resultado","brasil","dados","ações","processo","público","pública",
    # siglas UF...
    "UFG","UFMS","UFGD","UFMT","FURG","UFPel","UTFPR","UNIPAMPA","UFFS","UFV",
    "UNIFAL","UFJF","UFSJ","UFTM","UFF","UFMG","UNIFESP","UFU","UNIR","UFT",
    "UFAC","UFAM","UNIFESSPA","UFOPA","UFRR","UFRA","UFAL","UFS","UFCG","UFERSA",
    "UFRPE","UNIVASF","UFPI","UFC","UFCA","UNILAB","UFDPar","UFMA","UFRN","UFPB",
}

STOPWORDS = STOP_PT.union(EXTRA_SW)

nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner"])
nlp.max_length = 3_000_000

def spacy_tokenizer(text):
    doc = nlp(text)
    return [t.lemma_.lower() for t in doc
            if t.is_alpha and len(t) > 3
            and t.lemma_.lower() not in STOPWORDS]

vectorizer = CountVectorizer(tokenizer=spacy_tokenizer, stop_words=list(STOPWORDS), max_df=0.9)

# Redução de dimensionalidade

# Representação
representation_model = {
    "KeyBERT": KeyBERTInspired(),
    "MMR": MaximalMarginalRelevance(diversity=0.3),
}

# Loop de treinamento
N_NEIGHBORS = list(range(3, 15, 1))
CLUSTER_SIZES = list(range(10, 25, 1))
results = []
counter = 0
for n in N_NEIGHBORS:
    umap_model = UMAP(n_neighbors=n, random_state=SEED)
    for m in CLUSTER_SIZES:
        counter+=1
        hdbscan_model = HDBSCAN(min_cluster_size=m,
                                metric="euclidean",
                                cluster_selection_method="eom",
                                prediction_data=False)

        topic_model = BERTopic(embedding_model=emb_model,
                               umap_model=umap_model,
                               hdbscan_model=hdbscan_model,
                               vectorizer_model=vectorizer,
                               representation_model=representation_model,
                               top_n_words=10,
                               verbose=False)

        topics, _ = topic_model.fit_transform(docs_pp, embeddings)

        topic_words = [[w for w, _ in topic_model.get_topic(t)[:10]]
                       for t in topic_model.get_topics().keys() if t != -1]

        out_json = MODEL_DIR / f'topics_nneighbors_{n}_clustersize_{m}.json'
        json.dump(topic_words, open(out_json, 'w', encoding='utf-8'), ensure_ascii=False)

        results.append({'n_neighbors' :n,
                        'cluster_size': m,
                        'n_topics': len(topic_words),
                        'topics_path': str(out_json)})
        print(f'Results {counter}/{len(N_NEIGHBORS)*len(CLUSTER_SIZES)}: {results}')

pd.DataFrame(results).to_csv(MODEL_DIR / 'runs_catalog.csv', index=False)

