# Ontologue: Declarative Benchmark Construction for Ontological Multi-Label Classification

Ontologue is a toolkit for ontological multi-label classification dataset construction from DBPedia. This toolkit allows users to control contextual, distributional, and structured properties and create customized datasets. 

The codes and data in this respository aim to:
1. extract Wikipedia abstracts and the associated labels from the DBPedia ontology and create customized Hierarchical Multi-label (HMC) datasets.
2. analyze the customized datasets and the current HMC benchmarks in terms of their distribution, structure, and context.
3. provide 4 HMC benchmarks for future studies

## How To Use

In this section, we provide tutorial for Ontologue.

### Required Libraries

The code is written in python3 and jupyter notebook.

- Networkx
- tqdm
- [pyvis](https://pyvis.readthedocs.io/en/latest/install.html)
- [sentence_transformers](https://www.sbert.net/)
- [arff](https://pypi.org/project/arff/)
- numpy
- scipy


### Descriptions for Each File
**Extract_DBPedia.ipynb**: You can use this jupyter notebook to create customized datasets from DBPedia. Please see the annotations in the notebook for more instructions.
**Analyze_Dataset.ipynb**: You can use this notebook to analyze and visualize the customized datasets from Ontologue and the current HMC benchmarks.
**convert_medmentions.py**: This script was used to convert [MedMentions](https://github.com/chanzuckerberg/MedMentions) [data](https://github.com/chanzuckerberg/MedMentions/tree/master/full) to preferred data structure for Ontologuue
**input_data.py** and **utils.py** include helper functions.

### Data

To start the code from scratch, you will need to 

We also provide processed data on [Google Drive](https://drive.google.com/drive/folders/1Y1QHfy6fEAxuz4XGhnNHxl130cGoXlZb?usp=sharing):
- processed_DBPedia.tar.gz




