# SANIHFNLPpt

### Author

Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

### Description

Sequentially Activated Neuronal Input hopfield natural language processing (SANIHFNLP) for PyTorch - experimental

### License

MIT License

### Installation
```
conda create -n anntf2 python=3.7
source activate anntf2
conda install nltk [required for tokenisation]
conda install spacy [required for tokenisation]
python3 -m spacy download en_core_web_md [required for tokenisation]
conda install networkx [required for Draw]
pip install matplotlib==2.2.3 [required for Draw]
pip install yattag [required for XML]
pip install torch [required for vectoriseComputation]
pip install torch_geometric [required for vectoriseComputation:PyG]
```

### Execution
```
source activate anntf2
python3 SANIHFNLPpt_main.py
```
