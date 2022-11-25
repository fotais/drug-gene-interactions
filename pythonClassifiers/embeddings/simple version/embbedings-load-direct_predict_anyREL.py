# Define Neo4j connections
from neo4j import GraphDatabase
import pandas as pd
from pykeen.triples import TriplesFactory

host = 'bolt://localhost:7687'
user = 'neo4j'
password = 'iasis'
driver = GraphDatabase.driver(host,auth=(user, password))


def run_query(query, params={}):
    with driver.session() as session:
        result = session.run(query, params)
        return pd.DataFrame([r.values() for r in result], columns=result.keys())

def getIdFromCUI(cui): 
    cquery = r"MATCH (s:Annotation) WHERE (s.id='"+cui+ r"') RETURN toString(id(s)) as id"
    loc_id = run_query(cquery)
    if len(loc_id)==0:
        print('no id for cui: '+cui)
    return loc_id['id'][0]

listEmbeddings = ['TransE', 'TransD', 'TransF', 'TransH', 'DistMult', 'RESCAL', 'HoLE']
listExaminedRelations = ['AFFECTS', 'INTERACTS_WITH', 'STIMULATES', 'INHIBITS', 'AUGMENTS']

#check1 = input("Press return to start:")

import torch
from pykeen.pipeline import pipeline_from_path

print('loading from the file to get link predictions.') 
model = torch.load(listEmbeddings[0]+'/trained_model.pkl')
tf=TriplesFactory.from_path_binary('dictionary.pt')

print('Done')
from pykeen.models.predict import predict_triples_df 
data = pd.read_csv("/home/fot/workspace/python/approachb-DTI-ALL-features-cuis-extended_test.csv")
testPairs=data["CUI_PAIR"]
pairs_ground=data["INTERACTS"]

import numpy as np

###Examine all relations in the list and keep the max score
pairs_pred={}
rows, cols = (len(testPairs), 3)
idList = [["" for i in range(cols)] for j in range(rows)]
j=0
ids_ground = {}
while (j<len(testPairs)):
    testPair=testPairs[j]
    cuiList = testPair.split("_");
    drCUI=cuiList[0]
    disCUI= cuiList[1]
    disId = getIdFromCUI (disCUI)
    drId = getIdFromCUI (drCUI)
    ids_ground[drId+'_'+disId]=int(pairs_ground[j])
    #print('drug CUI: ', drCUI, ' to drug id: ', drId)
    #print('disease CUI: ', disCUI, ' to disease id: ', disId)
    #print('groundtruth=' , ids_ground[drId+'_'+disId])
    idList[j][0]=drId
    idList[j][2]=disId
    j=j+1

r=-1
for rel in listExaminedRelations:
    r=r+1
    j=0
    while (j<len(testPairs)):
        idList[j][1]=rel
        j=j+1
    ###Get prediction for all pairs of testset  for a specific relation type
    df = predict_triples_df(model,triples=idList, triples_factory=tf) 
    ###Get min, max values to transform scores within the range [0,1]
    max = np.amax(df.loc[:,'score'])
    min = np.amin(df.loc[:,'score'])
    range = max-min
#    print ('max=', max, ', min=', min)
    i=0
    while (i<len(df)):
        pair=df.loc[i,'head_label']+"_"+df.loc[i,'tail_label']
        normalized_pred=(df.loc[i,'score']-min)/range
        if (pair in pairs_pred):
            if (pairs_pred[pair]<normalized_pred):
                pairs_pred[pair]=normalized_pred
        else:
            pairs_pred[pair]=normalized_pred
        i=i+1

###now convert to 2 lists in order to calculate accuracy
y_real=[0]*len(ids_ground)
y_pred=[0]*len(ids_ground)
k=0
for pair in pairs_pred:
    y_real[k]=ids_ground[pair]
    y_pred[k]=round(pairs_pred[pair])
    print('y_real: %f and y_pred: %f',y_real[k], y_pred[k])
    k=k+1

###Print classification report results
from sklearn.metrics import classification_report
print(classification_report(y_real, y_pred))
