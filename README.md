# Ontologue: Declarative Benchmark Construction for Ontological Multi-Label Classification

Ontologue is a toolkit for ontological multi-label classification dataset construction from DBPedia. This toolkit allows users to control contextual, distributional, and structured properties and create customized datasets. 

The codes and data in this respository aim to:
1. extract Wikipedia abstracts and the associated labels from the DBPedia ontology and create customized Hierarchical Multi-label (HMC) datasets.
2. analyze the customized datasets and the current HMC benchmarks in terms of their distribution, structure, and context.
3. provide four HMC benchmarks for future studies

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

### Data

To start the process from scratch, you will need to download necessary data from [DBPedia](https://databus.dbpedia.org/dbpedia/collections/dbpedia-snapshot-2021-09/), which include
1. [Wikipedia short abstract](https://databus.dbpedia.org/dbpedia/text/short-abstracts/2021.08.01/short-abstracts_lang=en.ttl.bz2)
2. [Ontology skos Graph](https://databus.dbpedia.org/dbpedia/generic/categories/2021.09.01/categories_lang=en_skos.ttl.bz2)
3. [Subject Lables](https://databus.dbpedia.org/dbpedia/generic/categories/2021.09.01/categories_lang=en_articles.ttl.bz2)

We also provide processed data `processed_DBPedia.tar.gz` on [Google Drive](https://drive.google.com/drive/folders/1Y1QHfy6fEAxuz4XGhnNHxl130cGoXlZb?usp=sharing):

The products of the proposed benchmarks (Engineering, Law, Comedy, and Main) from Ontologue are also provided on [Google Drive](https://drive.google.com/drive/folders/1Y1QHfy6fEAxuz4XGhnNHxl130cGoXlZb?usp=sharing)

The proposed benchmarks in arff format can be found with this [link](https://drive.google.com/file/d/1UbCMNltGkN4Fbhs070duSzjTTyJFmPcb/view?usp=sharing)


### Descriptions for Each File

- Extract_DBPedia.ipynb: You can use this jupyter notebook to create customized datasets from DBPedia. Please see the annotations in the notebook for more instructions.
- Analyze_Dataset.ipynb: You can use this notebook to analyze and visualize the customized datasets from Ontologue and the current HMC benchmarks.
- convert_medmentions.py: This script was used to convert [MedMentions](https://github.com/chanzuckerberg/MedMentions) [data](https://github.com/chanzuckerberg/MedMentions/tree/master/full) to required data structure for Ontologuue
- input_data.py: include helper functions.
- utils.py: include helper functions.

## Apply to Your Own Graph

We also show that Ontologue can be appied to a different source. [MedMentions](https://github.com/chanzuckerberg/MedMentions) provides annotations from UMLS on over 4k papers. We convert the annotations from MedMentions to the required format for Ontologue with `convert_medmentions.py`. You can modify the code to fit the data structure of your own source. 





