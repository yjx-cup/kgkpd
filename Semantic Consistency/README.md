## Welcome to use the KGKPD model.

### Description

`knowledge_graph_prep.py`  is used to convert ConceptNet to obtain prior knowledge.

`knowledge_graph.py`  is used to obtain the semantic consistency matrix.

### ConceptNet

The assertion lists (knowledge graphs) can be downloaded at

[https://github.com/commonsense/conceptnet5/wiki/Downloads](https://github.com/commonsense/conceptnet5/wiki/Downloads)

for this approach versions 5.7.0 or 5.5.0 are used.

### Struct

make sure the folder is structured correctly:

* Semantic Consistency
    * ConceptNet
        * Assertion55
        * Assertion57
    * Stored matrices
        * your_dataset_kg_(versions)_info.json
    * KG_crop_(versions).csv
    * KG_lookup_(versions).csv
    * knowledge_graph.py
    * knowledge_graph_prep.py

### Run
```
python knowledge_graph_prep.py
python knowledge_graph.py
```



