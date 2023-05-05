# Methods for Drug-Gene Interaction Prediction on the Biomedical Literature Knowledge Graph:

These projects provide the implementation of a set of methods for link prediction on a disease-specific Biomedical Literature Knowledge Graph. The preliminary pipeline for the creation of this Knowledge Graph (KG) from scientific literature and open databases is provided [here](https://github.com/tasosnent/iASiS_WP4_java_modules) [1].

## Licence & Required Citation
For any use of the current source code or the Full-DTIs-LC-Benchmark.csv file in your work, **a citation to the following paper is expected:**
*Aisopos, F. & Paliouras, G. (2023). Comparing Methods for Drug-Gene Interaction Prediction on the Biomedical Literature Knowledge Graph: Performance vs. Explainability. [https://www.researchsquare.com/article/rs-2791034/latest.pdf](https://www.researchsquare.com/article/rs-2791034/latest.pdf)*

NCSR Demokritos module Copyright 2021 Fotis Aisopos
The Java code and CSV file are provided **only for academic/research use and are licensed under the Apache License, Version 2.0 (the "License")**; you may not use this file except in compliance with the License. You may obtain a copy of the License at: https://www.apache.org/licenses/LICENSE-2.0 .


## AnyBURL
The folder AnyBURL_customized contains the classes that have been updated in the original Java project. To run this project, one has to run the provided customized jar file, along with the learning/apply configuration files as explained  [here](https://web.informatik.uni-mannheim.de/AnyBURL). 
* java -Xmx12G -cp AnyBURL-fot.jar de.unima.ki.anyburl.LearnReinforced config-learn.properties
* java -Xmx12G -cp AnyBURL-fot.jar de.unima.ki.anyburl.Apply config-apply.properties

AnyBURL requires as input a knowledge graph, in the form of tab separated values (tsv) files. Thus, the full Biomedical Literature Knowledge Graph has to be first extracted from Neo4j into a tsv. The groundtruth triples have to be divided into ten folds via ten different files. For each repetition, the nine folds must be merged with the main knowledge graph tsv file, and one has been kept as testset. 

When running the Apply step, the updated source Java code calculates the Precision, Recall and F1-Score metrics for each fold.

## SemaTyP 
Under SemaTyP_customized, we have re-implemented SemaTyP in Java, in order to make use of the Java API to the Neo4j database holding the KG. The Java SemaTyP implementation collects all DTD paths relating drugs with targets, and ignores article nodes, MENTIONED_IN relations, as well as triples retrieved only from a single article.

To run SemaTyP, one needs to run the main method of the *SemaTyP_Neo4JAlgorithms class*, providing as input parameters 
1. the path where features will be extracted
2. the path of the main folder of Neo4j database (/graph.db)
3. a groundtruth flag ("1" for extracting positive pairs' features and "0" for extracting negative pairs' features)

e.g. "java -Xmx12g -jar ./SemaTyP_Neo4JAlgorithms.jar ./sematyp/ /home/user//workspace/neo4j-community-3.5.26/data/databases/graph.db 1"

After constructing the features csv file, the classification task can be ran in python, using the *SemaTyP-10foldCV-LR-DTIs_drug-gene.py* file under /pythonClassifiers/white-box-methods/ .

## BLGPA
This is an extension of the DDI-BLKG method [2]. The path collection and SE+PR feature extraction modules have been also implemented in Java under the BioGraphPath/ folder, exploiting the Java API to the Neo4j database. 

To run BLGPA, one needs to first run the main method of the *Enriched_DTI_BLKG class*, providing as input parameters: 
1. the path where features will be extracted
2. the path of the main folder of Neo4j database (/graph.db) 
3. a groundtruth flag ("1" for extracting positive pairs' features and "0" for extracting negative pairs' features)

e.g. "java -Xmx12g -jar ./Enriched_DTI_BLKG.jar ./blgpa/ /home/user//workspace/neo4j-community-3.5.26/data/databases/graph.db 1"

After constructing the features csv file, the classification task can be ran in python, using the *BLGPA_random-forest-undersampling-10foldCV_inner5foldCV-DTIs-approachB.py* file under /pythonClassifiers/embeddings . In case of feature selection, the *BLGPA_random-forest-undersampling-10foldCV_inner5foldCV-DTIs-approachB_FeatureSelection.py* file has to be ran.

## Graph Embeddings
The various embedding models (TransE, DisMult, HoLE and RESCAL) can be ran sequencially using the *neo4jembbedings_loadANDclassifyTensors.py* file under /pythonClassifiers/embeddings. This file requires a connection with the Neo4j service, as well as a file providing the groundtruth of the DTIs golden standard in the variable data.
Then the code retrieves for every pair of CUIs, the groundtruth flag ("1" or "0"):
cuiPairs=data["CUI_PAIR"]
pairs_ground=data

The final lines provide a report of the performance of each model, in terms of macro average of Precision, Recall and F1-score.

## RGCN
For implementing the RGCN classifier, the Neo4j KG has to be extracted in a tsv file, in order to build the Graph Convolution Network out of it. This has to be divided into the following files:
entities.tsv
relations.tsv
relations_entities.tsv
relations_articles.tsv
articles.tsv
to represent the different type of entities and relations of the KG.

Then the *pytorch-geometric_GCN.py* can be ran, under /drug-gene-interactions/pythonClassifiers/GCN.
Within this file, the path of the aforementioned tsv files, as well as of the  positive and negative groundtruth tsv files has to be provided.

The final lines provide a report of the performance of each model, in terms of macro average of Precision, Recall and F1-score.


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
