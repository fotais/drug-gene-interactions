# Define Neo4j connections
from neo4j import GraphDatabase
import pandas as pd
from sklearn import metrics
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import torch
from pykeen.pipeline import pipeline_from_path
from pykeen.models.predict import predict_triples_df 
import numpy as np
from sklearn.model_selection import KFold
from numpy import mean
from numpy import std
from imblearn.under_sampling import RandomUnderSampler
import csv
import os

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

listEmbeddings = ['HoLE', 'TransE', 'DistMult', 'RESCAL']

print('First prepare the cross-validation procedure.')
cv = KFold(n_splits=10, random_state=1, shuffle=True)

precision = np.zeros(10)
recall = np.zeros(10)
f1=np.zeros(10)
f=0

data = pd.read_csv("/home/faisopos/workspace/python/approachb-DTI-ALL-features-cuis-extended.csv")
#cuiPairs=data["CUI_PAIR"]
#pairs_ground=data["INTERACTS"]
x=data["CUI_PAIR"]
y=data["INTERACTS"]
#initial data ratio
undersample = RandomUnderSampler(sampling_strategy=0.1)
x, pairs_ground = undersample.fit_resample(x.values.reshape(-1,1), y)
cuiPairs0=pd.DataFrame(x)
cuiPairs=cuiPairs0[0]

#print('cuiPairs', cuiPairs[0])

print('Start 10 CV...')
for train_index, test_index in cv.split(cuiPairs):
    print("Run ", f+1)

    X_trainPairs, X_testPairs= cuiPairs.iloc[train_index], cuiPairs.iloc[test_index] 
    y_train, y_test = pairs_ground.iloc[train_index], pairs_ground.iloc[test_index]

    print('len=', len(X_trainPairs)) 
    #positive train pair must be inserted into KG before embedding creation...
    #j=0

    #save train and test pairs for this fold in CSV
    embedFoldPath = listEmbeddings[0]+'/'+listEmbeddings[0]+str(f)
    os.makedirs(os.path.dirname(embedFoldPath+'/training.csv'), exist_ok=True)
    f_tr = open(embedFoldPath+'/training.csv', 'w')
    f_tes= open(embedFoldPath+'/testing.csv', 'w')
    # create the csv writer
    writerTr = csv.writer(f_tr, delimiter=',')
    writerTe = csv.writer(f_tes)

    for j in X_trainPairs.keys():
        interaction = y_train[j]
        #print('j='+str(j)+' , interaction='+str(interaction))
        # write a row to the csv file
        #writerTr.writerow(['aa bb',str(y_train[j])]) #str(X_trainPairs[j])
        writerTr.writerow([str(X_trainPairs[j]),str(y_train[j])]) 

        if interaction==1:
           trainPair=X_trainPairs[j]
           cuiList = trainPair.split("_");
           drCUI=cuiList[0]
           tarCUI= cuiList[1]
           #print('INSERT INTO NEO4J INTERACTION BETWEEN:', drCUI, tarCUI)
           #check0 = input("PAUSE.")
           insert = run_query("""
               MATCH (s:Entity),(t:Entity) WHERE s.id = '"""+drCUI+"""' AND t.id = '"""+tarCUI+"""'
               CREATE (s)-[r:INTERACTION]->(t) RETURN type(r)
               """)
    
    data = run_query("""MATCH (s)-[r]->(t)  RETURN toString(id(s)) as source, toString(id(t)) AS target, type(r) as type """)

    print('Importing neo4j to TriplesFactory...')

    tf1 = TriplesFactory.from_labeled_triples(
      data[["source", "type", "target"]].values,
      create_inverse_triples=False,
      entity_to_id=None,
      relation_to_id=None,
      compact_id=False,
      filter_out_candidate_inverse_relations=True,
      metadata=None,
    )
    #change this for every embedding
    tf1.to_path_binary('dictionary_HoLE.pt')
    print('Done! ')

    ###### OK NOW DELETE EXTRA INTERACTIONS FROM NEO4J...
    print('OK now remove groundtruth interactions from Neo4j.')
    j=0
    for j in X_trainPairs.keys():
        interaction = y_train[j]
        if interaction==1: #positive train pair must be deleted from KG  after embedding creation...
            trainPair=X_trainPairs[j]
            cuiList = trainPair.split("_");
            drCUI=cuiList[0]
            tarCUI= cuiList[1]
            delete = run_query("""
                MATCH (s:Entity)-[r:INTERACTION]->(t:Entity) WHERE s.id = '"""+drCUI+"""' AND t.id = '"""+tarCUI+"""'
                DELETE r
                """)
    f_tr.close()
    #check1 = input("PAUSE.")
    
    training, testing, validation = tf1.split([.8, .1, .1])

#    check1 = input("Kill neo4j and Press return/..")

    print('Now preparing embedding for this fold...')

    for embed in listEmbeddings:
        print('training ', embed)
        result = pipeline(
            training=training,
            testing=testing,
            validation=validation,
            model=embed,
            stopper='early',
            epochs=100,
            dimensions=100,
            random_seed=42,
            device='cuda:0'
        )
        print(embed, ': Saving created model to a local file...')
        result.save_to_directory(embed+'/'+embed+str(f))
        break;

#    check2 = input("Start neo4j and Press return...")


    ################second part: load 

#    check3 = input("Finally kill again neo4j and Press return...")

    print('loading from the file to get link predictions.') 
    model = torch.load(embed+'/'+embed+str(f)+'/trained_model.pkl')
    tf=TriplesFactory.from_path_binary('dictionary_'+embed+'.pt')

    print('Done')
    
    p=0
    ids_ground = {}
    rows=len(X_testPairs)
    cols = 3
    idList = [["","",""] for r in range(rows)]
    for j in X_testPairs.keys():
        # write a row to the csv file
        writerTe.writerow([str(X_testPairs[j]),str(y_test[j])])
        
        testPair=X_testPairs[j]
        cuiList = testPair.split("_");
        drCUI=cuiList[0]
        disCUI= cuiList[1]
        disId = getIdFromCUI (disCUI)
        drId = getIdFromCUI (drCUI)
        ids_ground[drId+'_'+disId]=int(y_test[j])
#        print('drug CUI: ', drCUI, ' to drug id: ', drId)
#        print('target CUI: ', disCUI, ' to target id: ', disId)
#        print('groundtruth=' , ids_ground[drId+'_'+disId])
        idList[p][0]=drId
        idList[p][1]='INTERACTION'
        idList[p][2]=disId
        p=p+1


    ###Get prediction for all pairs of testset  for a specific relation type
    df = predict_triples_df(model,triples=idList, triples_factory=tf) 

    ###Get min, max values to transform scores within the range [0,1]
    mmax = np.amax(df.loc[:,'score'])
    mmin = np.amin(df.loc[:,'score'])
    mrange = mmax-mmin
#    print ('max=', mmax, ', min=', mmin)

    pairs_pred={}
    i=0
    while (i<len(df)):
        pair=df.loc[i,'head_label']+"_"+df.loc[i,'tail_label']
#        print(pair,',',df.loc[i,'score'])
        pairs_pred[pair]=(df.loc[i,'score']-mmin)/mrange
        i=i+1

    ###now convert to 2 lists in order to calculate accuracy
    y_real=[0]*len(ids_ground)
    y_pred=[0]*len(ids_ground)
    k=0
    for pair in pairs_pred:
        y_real[k]=ids_ground[pair]
        y_pred[k]=round(pairs_pred[pair])
        k=k+1

    precision[f]=metrics.precision_score(y_real, y_pred)
    recall[f]=metrics.recall_score(y_real, y_pred)
    f1[f]=metrics.f1_score(y_real, y_pred)
    f=f+1
    # close the pair csv files of this fold...

    f_tes.close()

print("10-fold CV finished...calculating macro avgs:")
   
# report performance
print('Precision: %.3f (%.3f)' % (mean(precision), std(precision)))
print('recall: %.3f (%.3f)' % (mean(recall), std(recall)))
print('f1-score: %.3f (%.3f)' % (mean(f1), std(f1)))
