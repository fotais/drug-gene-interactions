# Methods for Drug-Gene Interaction Prediction on the Biomedical Literature Knowledge Graph:

These projects provide the implementation of a set of methods for link prediction on a disease-specific Biomedical Literature Knowledge Graph. The preliminary pipeline for the creation of this Knowledge Graph (KG) from scientific literature and open databases is provided [here](https://github.com/tasosnent/iASiS_WP4_java_modules) [1].

## Licence & Required Citation
For any use of the current source code or the Full-DTIs-LC-Benchmark.csv file in your work, **a citation to the following paper is expected:**
*Aisopos, F. & Paliouras, G. (2023). Comparing Methods for Drug-Gene Interaction Prediction on the Biomedical Literature Knowledge Graph: Performance vs. Explainability. [https://www.researchsquare.com/article/rs-2791034/latest.pdf](https://www.researchsquare.com/article/rs-2791034/latest.pdf)*

NCSR Demokritos module Copyright 2021 Fotis Aisopos
The Java code and CSV file are provided **only for academic/research use and are licensed under the Apache License, Version 2.0 (the "License")**; you may not use this file except in compliance with the License. You may obtain a copy of the License at: https://www.apache.org/licenses/LICENSE-2.0 .


## AnyBURL
The folder AnyBURL_customized contains the classes that have been updated in the original Java project. To run this project, one has to run the provided customized jar file, along with the learning/apply configuration files as explained  [here](https://web.informatik.uni-mannheim.de/AnyBURL). 
AnyBURL requires as input a knowledge graph, in the form of tab separated values (tsv) files. Thus, the full Biomedical Literature Knowledge Graph has to be first extracted from Neo4j into a tsv. The groundtruth triples have to be divided into ten folds via ten different files. For each repetition, the nine folds must be merged with the main knowledge graph tsv file, and one has been kept as testset. When running the Apply step, the updated source Java code calculates the Precision, Recall and F1-Score metrics for each fold.

## SemaTyP 
Under SemaTyP_customized, we have re-implemented SemaTyP in Java, in order to make use of the Java API to the Neo4j database holding the KG. The Java SemaTyP implementation collects all DTD paths relating drugs with targets, and ignores article nodes, MENTIONED_IN relations, as well as triples retrieved only from a single article.
To run SemaTyP, one needs to run the main method of the SemaTyP_Neo4JAlgorithms class, providing as input parameters the path where features will be extracted, the path of the main folder of Neo4j database (/graph.db) and a groundtruth flag ("1" for extracting positive pairs' features and "0" for extracting negative pairs' features).

## BLGPA
This is an extension of the DDI-BLKG method [2]. The path collection and SE+PR feature extraction modules have been also implemented in Java exploiting the Java API to the Neo4j database. The random forest classifier has been built in python using the scikit-learn1 library. 
The code includes the CreateTargetMappings class (main class), and the auxiliary TargetEntry class, representing a target object with the various properties.
To run the aforementioned Java project, it is obvious that we need to have access to the following sources:
- TTD (to download the targets' information file in raw format)
- Entrez Programming Utilities (E-utilities) API (query PUG for PubChem ids and obtain a token to query for a TGT and an API key)
and also include needed jar libraries in the CLASSPATH.

## Graph Embeddings
We have used the PyKEEN library2 and PyTorch3 to produce the TransE, DisMult, HoLE and RESCAL graph embeddings.

## RGCN
For implementing the RGCN classifier, the KG had to be extracted in a tsv file, in order to build the Graph Convolution Network out of it. For this purpose, we have employed the RGCNEncoder of the PyTorch Geometric library4.

## Experimenting: Methods hyper-parameters

The following table provides an overview of the hyper-parameter values used in every method:

| AnyBURL                              | SemaTyP                                              | BLGPA | Embeddings |  RGCN |
| ------------------------------------ | ---------------------------------------------------- | ----- | ---------- | ----- |
| UN-SEEN NEGATIVE EXAMPLES=1          | Maximum path lengths=3                             |  Maximum path lengths=3     | Εmbedding size=100  | Encoder hidden layers=100 |
| TOP K OUTPUT=500                     | Logistic Regression parameters: penalty=L2, λ2=1.0, solver='lbfgs', max_iter=13000             |  Top-ranking paths=100     |  Max epoxhs=100 (early stop option)  | Decoder=DistMult |
| THRESHOLD CORRECT PREDICTIONS=5      |    |   Random Forest model parameters: no. of estimators=100, criterion=”gini”, max_depth=None, min_samples_split=2, min_samples_leaf=1, Feature Selection= SelectFromModel (threshold=0.003)    |  Random Forest model parameters: no. of estimators=100, criterion=”gini”, max_depth=None, min_samples_split=2, min_samples_leaf=1  | Optimizer = Adam optimization (learning_rate=0.01) |
| SNAPSHOTS_AT = 5000                  |  |  |  | Max epochs=15 / 50  (applied for 1:10 / 1:54 ratios respectively) |

   
       
## References
[1]:  Nentidis, A., Bougiatiotis, K., Krithara, A., & Paliouras, G. (2020, July). iasis open data graph: Automated semantic integration of disease-specific knowledge. In 2020 IEEE 33rd International Symposium on Computer-Based Medical Systems (CBMS) (pp. 220-225). IEEE.

[2]: Bougiatiotis, K., Aisopos, F., Nentidis, A., Krithara, A., & Paliouras, G. (2020). Drug-drug interaction prediction on a biomedical literature knowledge graph. In Artificial Intelligence in Medicine: 18th International Conference on Artificial Intelligence in Medicine, AIME 2020, Minneapolis, MN, USA, August 25–28, 2020, Proceedings 18 (pp. 122-132). Springer International Publishing.
