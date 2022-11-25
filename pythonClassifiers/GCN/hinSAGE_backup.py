import json
import pandas as pd
import numpy as np
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.metrics import mean_absolute_error, mean_squared_error

import stellargraph as sg
from stellargraph.mapper import HinSAGELinkGenerator
from stellargraph.layer import HinSAGE, link_regression
from tensorflow.keras import Model, optimizers, losses, metrics

import multiprocessing
from stellargraph import datasets
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from stellargraph.connector.neo4j import Neo4jGraphSAGENodeGenerator, Neo4jStellarGraph
from stellargraph import StellarGraph
from py2neo.data import Node, Relationship, Subgraph
import py2neo




# Create the Neo4j Graph database object; the parameters can be edited to specify location and authentication
neo4j_graph = py2neo.Graph(host="127.0.0.1", port=7687, user="neo4j", password="iasis")
#print neo4j stats
num_nodes = len(neo4j_graph.nodes)
num_relationships = len(neo4j_graph.relationships)
print("num_nodes", num_nodes, "num_relationships ", num_relationships )

#FOT: Query to retrieve neo4j (groundtruth???) into a dataFrame of labelled edges!
labelled_edges = neo4j_graph.run(
           """
           MATCH (s:Entity) -[r]-> (t:Entity)
           WHERE ((ANY(item IN s.sem_types WHERE item ='orch')) AND (ANY(item IN t.sem_types WHERE item ='gngm')))
           RETURN s.id AS source, t.id AS target, CASE WHEN type(r)="INTERACTS_WITH"  THEN 1 ELSE 0 END AS groundtruth 
           """
).to_data_frame()
labelled_edges=labelled_edges.drop_duplicates(subset=['source', 'target'])

#FOT: This must query the full Neo4j Graph
neo4j_df = neo4j_graph.run(
           """
           MATCH (s:Entity) -[r]-> (t:Entity)
           WHERE (ANY(item IN s.sem_types WHERE item ='orch')) AND (ANY(item IN t.sem_types WHERE item ='gngm'))  
           RETURN s.id AS source, r.subject_score[0] AS subfeature, t.id AS target, r.object_score[0] AS objfeature, type(r) AS rtype
           """
).to_data_frame()

#square_foo = pd.DataFrame(index=["a"])
#square_bar = pd.DataFrame(    {"y": [0.4, 0.1, 0.9], "z": [100, 200, 300]}, index=["b", "c", "d"])

sourceNodes=neo4j_df.filter(["source", "subfeature"])
sourceNodes=sourceNodes.drop_duplicates(subset=['source'])
sourceNodes=sourceNodes.set_index('source')
targetNodes=neo4j_df.filter(["target", "objfeature"])
targetNodes=targetNodes.drop_duplicates(subset=['target'])
targetNodes=targetNodes.set_index('target')
multiType_edges=neo4j_df.filter(["source", "target", "rtype"])

#print(targetNodes)

hetereogeneous_edges = StellarGraph({"source": sourceNodes, "target": targetNodes}, edges=multiType_edges, edge_type_column="rtype")
#hetereogeneous_edges = StellarGraph(sourceNodes, edges=labelled_edges, edge_type_column="label")

print("FINISHED WITH NEO4j SUCCESSFULLY!!!!!!!!!!")
#neo4j_sg = Neo4jStellarGraph(neo4j_graph, node_label="Entity")

#hinSAGE training configuration
batch_size = 100
epochs = 2
# Use 70% of edges for training, the rest for testing:
train_size = 0.9
test_size = 0.1


##G: whole graph  //  edges_with_ratings: groundtruth?
G= hetereogeneous_edges
print(G.info())
edges_with_ratings = labelled_edges  #dataset.load()


#split train-test set
edges_train, edges_test = model_selection.train_test_split(
    edges_with_ratings, train_size=train_size, test_size=test_size
)
edgelist_train = list(edges_train[["source", "target"]].itertuples(index=False))
edgelist_test = list(edges_test[["source", "target"]].itertuples(index=False))

#FOT: This must be numeric (1/0) groundtruth column!
labels_train = edges_train["groundtruth"]
labels_test = edges_test["groundtruth"]

#hinSAGE model generation...
num_samples = [8, 4]
generator = HinSAGELinkGenerator(
    G, batch_size, num_samples, head_node_types=["source", "target"]
)

train_gen = generator.flow(edgelist_train, labels_train, shuffle=True)
test_gen = generator.flow(edgelist_test, labels_test)

generator.schema.type_adjacency_list(generator.head_node_types, len(num_samples))
generator.schema.schema

check1=input("pause1")

hinsage_layer_sizes = [32, 32]
assert len(hinsage_layer_sizes) == len(num_samples)

hinsage = HinSAGE(
    layer_sizes=hinsage_layer_sizes, generator=generator, bias=True, dropout=0.0
)


x_inp, x_out = hinsage.in_out_tensors()

# Final estimator layer
score_prediction = link_regression(edge_embedding_method="concat")(x_out)


import tensorflow.keras.backend as K

def root_mean_square_error(s_true, s_pred):
    return K.sqrt(K.mean(K.pow(s_true - s_pred, 2)))


model = Model(inputs=x_inp, outputs=score_prediction)
model.compile(
    optimizer=optimizers.Adam(lr=1e-2),
    loss=losses.mean_squared_error,
    metrics=[root_mean_square_error, metrics.mae],
)

model.summary()

check4 = input("pause4...")

# Specify the number of workers to use for model training
num_workers = 4


history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=epochs,
    verbose=1,
    shuffle=False,
    use_multiprocessing=False,
    workers=num_workers,
)

test_metrics = model.evaluate(
    test_gen, use_multiprocessing=False, workers=num_workers, verbose=1
)

print("Test Evaluation:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

y_true = labels_test
# Predict the rankings using the model:
y_pred = model.predict(test_gen)

##FOT: round predictions to produce classification metrics: Pre, Rec, F1
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
print("\nModel Test set metrics:")
print("\troot_mean_square_error = ", rmse)
print("\tmean_absolute_error = ", mae)    

