import pandas as pd
import numpy as np

import stellargraph as sg
from tensorflow.keras import Model, optimizers, losses, metrics
import tensorflow.keras.backend as K

from collections import Counter
from statistics import mean
import multiprocessing
from stellargraph import StellarGraph
from py2neo.data import Node, Relationship, Subgraph
import py2neo

from stellargraph.layer import LinkEmbedding, link_classification
import tensorflow as tf
from stellargraph.mapper import HinSAGELinkGenerator
from stellargraph.layer import HinSAGE
from sklearn.model_selection import StratifiedKFold

import gc
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.metrics as tfm

from imblearn.under_sampling import RandomUnderSampler

class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        K.clear_session()

class Precision(tfm.Precision):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(Precision, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(Precision, self).update_state(y_true, y_pred, sample_weight)


class Recall(tfm.Recall):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(Recall, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(Recall, self).update_state(y_true, y_pred, sample_weight)

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1", from_logits=False, **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision(from_logits)
        self.recall = Recall(from_logits)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return (2 * p * r) / (p + r + tf.keras.backend.epsilon())

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

###################################################
###############k-fold Cross Validation#############
###################################################
#

tf.compat.v1.enable_eager_execution()
print(tf.config.list_logical_devices())

tf.debugging.set_log_device_placement(True)

#df = pd.read_csv("/home/faisopos/workspace/DTI-Prediction/FeatureExtraction/DTI-enriched_sematyp_features_test.csv")
df = pd.read_csv("DTI-enriched_sematyp_features.csv")

df_filtered=df.filter(["Drug_Target","GROUNDTRUTH"])
#####change initial sample pos-neg ratio
X=df_filtered["Drug_Target"]
y=df_filtered["GROUNDTRUTH"]
undersample = RandomUnderSampler(sampling_strategy='majority')
X, y = undersample.fit_resample(X.values.reshape(-1,1),y)
df_filtered2= pd.DataFrame({'pairs': list(X), 'GROUNDTRUTH': list(y)}, columns=['pairs', 'GROUNDTRUTH'])
df_filtered2 = df_filtered2.astype({"pairs": str})
df_filtered2['pairs']=df_filtered2["pairs"].str[2:19]
df_filtered2[['source', 'target']] = df_filtered2['pairs'].str.split('_', 1, expand=True)
df_filtered2 =df_filtered2.filter(['source', 'target', 'GROUNDTRUTH'])
groundtruth =df_filtered2
groundtruth.columns =["source", "target", "groundtruth"]
#####done
                 
# use k for k-fold CV:
k=2
VALIDATION_ACCURACY = []
VALIDATION_PRECISION = []
VALIDATION_RECALL= []
VALIDATION_FSCORE= []

skf = StratifiedKFold(n_splits = k, random_state = 1, shuffle = True)

for train_index, test_index in skf.split(groundtruth["source"], groundtruth["groundtruth"]):


###################################################
###Neo4j retrieval and import in StellarGraph######
###################################################


    # Create the Neo4j Graph database object; the parameters can be edited to specify location and authentication
    neo4j_graph = py2neo.Graph(host="192.168.11.52", port=7687, user="neo4j", password="iasis")

    sourceNodes = neo4j_graph.run(
        """
        MATCH (s:Entity)
        WHERE (NOT (ANY (item in s.sem_types WHERE item='gngm')))
        RETURN distinct s.id as source, substring(s.id,1,7) AS subfeature
        """
    ).to_data_frame()

    targetNodes = neo4j_graph.run(
        """
        MATCH (s:Entity)
        WHERE (ANY (item in s.sem_types WHERE item='gngm'))
        RETURN distinct s.id as target, substring(s.id,1,7) AS objfeature
        """
    ).to_data_frame()
    multiType_edges = neo4j_graph.run(
       """
       MATCH (s:Entity) -[r:INTERACTS_WITH]- (t:Entity)
       WHERE (ANY (item in t.sem_types WHERE item='gngm'))
       RETURN distinct s.id as source, type(r) AS rtype, t.id as target
       """
    ).to_data_frame()

    neoInsertList= pd.DataFrame(columns=["source","rtype", "target"])
    for i in train_index:
        if groundtruth.iloc[i]["groundtruth"]==1:
            src= groundtruth.iloc[i]["source"]
            trg= groundtruth.iloc[i]["target"]
            neoInsertList.append([src, "INTERACTS_WITH", trg])

    print(multiType_edges.size)        
    multiType_edges=multiType_edges.append(neoInsertList, ignore_index=True)

#    otherNodes = neo4j_graph.run(
#        """
#        MATCH (s:Entity)
#        WHERE ((NOT ANY (item in s.sem_types WHERE item='gngm')) AND (NOT ANY (item in s.sem_types WHERE item='phsu'))AND (NOT ANY (item in s.sem_types WHERE item='orch')))
#        RETURN distinct s.id as other, substring(s.id,1,7) AS objfeature
#        """
#    ).to_data_frame()

    print(multiType_edges.size)
    neoInsertList2 = neo4j_graph.run(
       """
       MATCH (s:Entity) -[r:ASSOCIATED_WITH]- (t:Entity)
       WHERE (ANY (item in t.sem_types WHERE item='gngm'))
       RETURN distinct s.id as source, type(r) AS rtype, t.id as target
       """
    ).to_data_frame()
    multiType_edges=multiType_edges.append(neoInsertList2, ignore_index=True)
    neoInsertList3 = neo4j_graph.run(
       """
       MATCH (s:Entity) -[r:TREATS]- (t:Entity)
       WHERE (ANY (item in t.sem_types WHERE item='gngm'))
       RETURN distinct s.id as source, type(r) AS rtype, t.id as target
       """
    ).to_data_frame()
    multiType_edges=multiType_edges.append(neoInsertList3, ignore_index=True)
    neoInsertList4 = neo4j_graph.run(
       """
       MATCH (s:Entity) -[r:USES]- (t:Entity)
       WHERE (ANY (item in t.sem_types WHERE item='gngm'))
       RETURN distinct s.id as source, type(r) AS rtype, t.id as target
       """
    ).to_data_frame()
    multiType_edges=multiType_edges.append(neoInsertList4, ignore_index=True)
    neoInsertList5 = neo4j_graph.run(
       """
       MATCH (s:Entity) -[r:ISA]- (t:Entity)
       WHERE (ANY (item in t.sem_types WHERE item='gngm'))
       RETURN distinct s.id as source, type(r) AS rtype, t.id as target
       """
    ).to_data_frame()
    multiType_edges=multiType_edges.append(neoInsertList5, ignore_index=True)


    print(multiType_edges)
    print(multiType_edges.size)

    sourceNodes=sourceNodes.set_index('source')
    targetNodes=targetNodes.set_index('target')
#    otherNodes=otherNodes.set_index('other')

    G = StellarGraph({"source": sourceNodes, "target": targetNodes}, edges=multiType_edges, edge_type_column="rtype")

    #print(G.info())

    print("FINISHED WITH IMPORTING NEO4j TO STELARGRAPH SUCCESSFULLY...")


###################################################
################hinSAGE training configuration#####
###################################################

    batch_size =100
    epochs = 100
    num_samples = [8, 4]
    hinsage_layer_sizes = [32, 32]
    assert len(hinsage_layer_sizes) == len(num_samples)


    G_train=G

    try:
        with tf.device('device:GPU:0'):
            generator = HinSAGELinkGenerator(
                G_train, batch_size, num_samples, head_node_types=["source", "target"]
            )
            generator.schema.type_adjacency_list(generator.head_node_types, len(num_samples))
            generator.schema.schema

            edges_train = groundtruth.iloc[train_index]
            edges_test = groundtruth.iloc[test_index]
            
            edgelist_train = list(edges_train[["source", "target"]].itertuples(index=False))
            edgelist_test = list(edges_test[["source", "target"]].itertuples(index=False))
            edgelabels_train = list(edges_train[["groundtruth"]].itertuples(index=False))
            edgelabels_test = list(edges_test[["groundtruth"]].itertuples(index=False))

            train_flow = generator.flow(edgelist_train, edgelabels_train, shuffle=True)
            test_flow = generator.flow(edgelist_test, edgelabels_test)

        #    check1=input("Press <ENTER> to create HinSAGE GCN...")

            hinsage = HinSAGE(
                layer_sizes=hinsage_layer_sizes, generator=generator, bias=True, dropout=0.0
            )
            x_inp, x_out = hinsage.in_out_tensors()

        #    check1=input("Press <ENTER> to check if tensorflow is using GPU:")

           # tf.print(tf.__version__)

            tf.debugging.set_log_device_placement(True)

#            prediction = LinkEmbedding(activation="sigmoid", method="concat")(x_out)
#            prediction = tf.keras.layers.Reshape((-1,))(prediction)
            prediction = link_classification(output_dim=1, output_act="sigmoid", edge_embedding_method='ip')(x_out)

            print(tf.config.list_physical_devices('GPU'))

            model = tf.keras.Model(inputs=x_inp, outputs=prediction)

#            check1=input("Press <ENTER> to compile Keras model...")
            model.compile(
                #optimizer=tf.keras.optimizers.Adam(lr=0.05),
                #loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, decay=0.1, momentum=0.1, nesterov=False),
                loss=tf.keras.losses.binary_crossentropy,
                metrics=["binary_accuracy", F1Score(name='f1'),Precision(name='precision'), Recall(name='recall')],
                run_eagerly=True
            )

            #model.summary()
            num_workers =5 

 #           check3=input("Compiled! Press <ENTER> to fit (train) model...")

            model.fit(train_flow, epochs=epochs, validation_data=test_flow, verbose=1, shuffle=False, use_multiprocessing=False, workers=num_workers, callbacks=ClearMemory())
  
            print("Now evaluating with test flow... ")
            results = model.evaluate(test_flow, use_multiprocessing=False, workers=num_workers, verbose=1)
            print("DONE!")
            results = dict(zip(model.metrics_names,results))

            bin_acc = results['binary_accuracy']
            precision = results['precision']
            recall = results['recall'] 
            f_score= results['f1']
            VALIDATION_ACCURACY.append(bin_acc)
            VALIDATION_PRECISION.append(precision)
            VALIDATION_RECALL.append(recall)
            VALIDATION_FSCORE.append(f_score)
            print("f1 score found in this fold: ", f_score)

            tf.keras.backend.clear_session()
    except RuntimeError as e:
        print(e)


###################################################
##########Print marco-average of metric values#####
###################################################
                
print("\nTest Set Metrics of the trained model:")
print("Average Accuracy: ", mean(VALIDATION_ACCURACY))
print("Average Precision: ", mean(VALIDATION_PRECISION))
print("Average Recall: ", mean(VALIDATION_RECALL))
print("Average F1-score: ", mean(VALIDATION_FSCORE))

