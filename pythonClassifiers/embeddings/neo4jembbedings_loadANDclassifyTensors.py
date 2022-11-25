# Define Neo4j connections
from neo4j import GraphDatabase
import pandas as pd
from sklearn import metrics
from pykeen.triples import TriplesFactory
import torch
import numpy as np
from sklearn.model_selection import KFold
from numpy import mean
from numpy import std
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

from typing import List
import pykeen.nn
from torch.autograd import Variable



host = 'bolt://localhost:7687'
user = 'neo4j'
password = 'iasis'
driver = GraphDatabase.driver(host,auth=(user, password))


def run_query(query, params={}):
    with driver.session() as session:
        result = session.run(query, params)
        return pd.DataFrame([r.values() for r in result], columns=result.keys())

def getIdFromCUI(cui): 
    cquery = r"MATCH (s:Entity) WHERE (s.id='"+cui+ r"') RETURN toString(id(s)) as id"
    loc_id = run_query(cquery)
    if len(loc_id)==0:
        print('no id for cui: '+cui)
    return loc_id['id'][0]

listEmbeddings = ['TransE', 'HoLE', 'DistMult', 'RESCAL']
embed = listEmbeddings[0]


from pykeen.triples import TriplesFactory

###########LOAD AND RUN k-FOLD CV

print('loading from the file to get link predictions.') 
model = torch.load(embed+'/trained_model.pkl')
tf=TriplesFactory.from_path_binary(embed+'_dictionary.pt')

print('Done')

relID=tf.relation_to_id["INTERACTS_WITH"]
relation: torch.IntTensor = torch.as_tensor(relID, device='cuda:1')
relation_representation_modules: List['pykeen.nn.Representation']  = model.relation_representations
relation_embeddings: pykeen.nn.Embedding  = relation_representation_modules[0]
relation_embedding_tensor: torch.FloatTensor  = relation_embeddings(indices=relation)
rel_embedding = Variable(relation_embedding_tensor).cpu()
print('relation interacts_with embedding_tensor: ')
print(rel_embedding.data[:])
#print(entity_embedding_tensor)

#check3 = input("pause...")

data = pd.read_csv("/home/faisopos/workspace/python/approachb-DTI-ALL-features-cuis-extended.csv")
cuiPairs=data["CUI_PAIR"]
pairs_ground=data["INTERACTS"]

print('create RF Classifier...')
#Create a RF Classifier
classifier=RandomForestClassifier(n_estimators=100)

 
k=10
precision = np.zeros(k)
recall = np.zeros(k)
f1=np.zeros(k)
print('Prepare the cross-validation procedure.')
cv = KFold(n_splits=k, random_state=1, shuffle=True)
f=0

print('Start '+str(k)+'-fold CV...')
for train_index, test_index in cv.split(cuiPairs):
    print("Run ", f+1)

    c_trainPairs, c_testPairs= cuiPairs.iloc[train_index], cuiPairs.iloc[test_index] 
    y_train, y_test = pairs_ground.iloc[train_index], pairs_ground.iloc[test_index]

    #print('len=', len(c_trainPairs)) 
    undersample = RandomUnderSampler(sampling_strategy=0.2)
    c, y_train = undersample.fit_resample(c_trainPairs.values.reshape(-1,1), y_train)
    c_train0=pd.DataFrame(c)
    c_train=c_train0[0]
    print('Undersample training samples: ',Counter(y_train))

    print("Now get node embeddings for each pair")
    entity_representation_modules: List['pykeen.nn.Representation']  = model.entity_representations
    entity_embeddings: pykeen.nn.Embedding  = entity_representation_modules[0]

    rows, cols = (len(y_train), 3*len(rel_embedding))
    X_train = [[0 for i in range(cols)] for j in range(rows)]
    for i,trainPair in enumerate(c_train):
        cuiList = trainPair.split("_");
        drCUI=cuiList[0]
        tarCUI= cuiList[1]
        tarId = getIdFromCUI (tarCUI)
        drId = getIdFromCUI (drCUI)

        dr_entity: torch.IntTensor = torch.as_tensor(tf.entity_to_id[drId], device='cuda:1')
        dr_entity_embedding_tensor: torch.FloatTensor  = entity_embeddings(indices=dr_entity)
        dr_embedding = Variable(dr_entity_embedding_tensor).cpu()

        tar_entity: torch.IntTensor = torch.as_tensor(tf.entity_to_id[tarId], device='cuda:1')
        tar_entity_embedding_tensor: torch.FloatTensor  = entity_embeddings(indices=tar_entity)
        tar_embedding = Variable(tar_entity_embedding_tensor).cpu()
        X_train[i]=np.concatenate((dr_embedding, rel_embedding, tar_embedding), axis=None)#torch.cat((dr_embedding, rel_embedding, tar_embedding), 0)# np.concatenate((dr_embedding, rel_embedding, tar_embedding), axis=None)
        #if (i%30)==0:
        #    print(trainPair+':'+str(X_train[i]))#print(trainPair+':'+str(y_train[i])
        #    classifier.fit(X_train, y_train)


    print("classifier fit...")
    classifier.fit(X_train, y_train)

    print("Now get node embeddings for each TEST pair")
    rows, cols = (len(y_test), 3*len(rel_embedding))
    X_test = [[0 for i in range(cols)] for j in range(rows)]
    
    for i,testPair in enumerate(c_testPairs):
        cuiList = testPair.split("_");
        drCUI=cuiList[0]
        tarCUI= cuiList[1]
        tarId = getIdFromCUI (tarCUI)
        drId = getIdFromCUI (drCUI)

        dr_entity: torch.IntTensor = torch.as_tensor(tf.entity_to_id[drId], device='cuda:1')
        dr_entity_embedding_tensor: torch.FloatTensor  = entity_embeddings(indices=dr_entity)
        dr_embedding = Variable(dr_entity_embedding_tensor).cpu()

        tar_entity: torch.IntTensor = torch.as_tensor(tf.entity_to_id[tarId], device='cuda:1')
        tar_entity_embedding_tensor: torch.FloatTensor  = entity_embeddings(indices=tar_entity)
        tar_embedding = Variable(tar_entity_embedding_tensor).cpu()
        X_test[i]= np.concatenate((dr_embedding, rel_embedding, tar_embedding), axis=None)
        #print(testPair+':'+str(y_test.iloc[i]))
    
    print("prediction")
    y_pred = classifier.predict(X_test)
    precision[f]=metrics.precision_score(y_test, y_pred)
    recall[f]=metrics.recall_score(y_test, y_pred)
    f1[f]=metrics.f1_score(y_test, y_pred)
    f=f+1


print("10-fold CV finished...calculating macro avgs:")
   
# report performance
print('Precision: %.3f (%.3f)' % (mean(precision), std(precision)))
print('recall: %.3f (%.3f)' % (mean(recall), std(recall)))
print('f1-score: %.3f (%.3f)' % (mean(f1), std(f1)))
