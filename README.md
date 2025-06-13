# BERTopic-OCTIS PROFIAP

Topic modeling aplication in the abstracts of PROFIAP (Professional Master's in Public Administration) using **BERTopic** and evaluation metrics from **OCTIS**.


## Quick Start

```bash
# clone the repository
git clone https://github.com/<user>/bertopic-project.git
cd bertopic-project

# create a virtual environment and install dependencies
pyenv install 3.10.12
pyenv local 3.10.12
poetry install

# download spaCy Portuguese model
chmod +x scripts/download_spacy_model.sh
poetry run scripts/download_spacy_model.sh
```

## License

MIT
