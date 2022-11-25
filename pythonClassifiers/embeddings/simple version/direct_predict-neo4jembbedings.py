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

listEmbeddings = ['TransE', 'DistMult', 'RESCAL', 'HoLE']

print('First prepare the cross-validation procedure.')
cv = KFold(n_splits=10, random_state=1, shuffle=True)

precision = np.zeros(10)
recall = np.zeros(10)
f1=np.zeros(10)
f=0

data = pd.read_csv("/home/faisopos/workspace/python/approachb-DTI-ALL-features-cuis-extended.csv")
cuiPairs=data["CUI_PAIR"]
pairs_ground=data["INTERACTS"]

print('Start 10 CV...')
for train_index, test_index in cv.split(cuiPairs):
    print("Run ", f+1)

    X_trainPairs, X_testPairs= cuiPairs.iloc[train_index], cuiPairs.iloc[test_index] 
    y_train, y_test = pairs_ground.iloc[train_index], pairs_ground.iloc[test_index]

    #positive train pair must be inserted into KG before embedding creation...
    j=0
    for j in X_trainPairs.keys():
        interaction = y_train[j]
#        print('j='+str(j)+' , interaction='+str(interaction))
        if interaction==1:
           trainPair=X_trainPairs[j]
           cuiList = trainPair.split("_");
           drCUI=cuiList[0]
           tarCUI= cuiList[1]
#           print('INSERT INTO NEO4J INTERACTION BETWEEN:', drCUI, tarCUI)
           insert = run_query("""
               MATCH (s:Entity),(t:Entity) WHERE s.id = '"""+drCUI+"""' AND t.id = '"""+tarCUI+"""'
               CREATE (s)-[r:INTERACTION]->(t) RETURN type(r)
               """)
        #j=j+1
           
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
    tf1.to_path_binary('dictionary.pt')
    print('Done! ')

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
            device='cuda'
        )
        print(embed, ': Saving created model to a local file...')
        result.save_to_directory(embed+'/'+embed+str(f))
        break;

#    check2 = input("Start neo4j and Press return...")

    ###### AT THE END OF EACH FOLD, DELETE FROM NEO4J...
    print('OK now remove interactions from Neo4j.')
    j=0
    for j in X_trainPairs.keys():
        interaction = y_train[j]
        if interaction==1: #positive train pair must be deleted from KG  after embedding creation...
            trainPair=X_trainPairs[j]
            cuiList = trainPair.split("_");
            drCUI=cuiList[0]
            tarCUI= cuiList[1]
#            print('DELETE FROM NEO4J INTERACTION BETWEEN:', drCUI, tarCUI)
            delete = run_query("""
                MATCH (s:Entity)-[r:INTERACTION]->(t:Entity) WHERE s.id = '"""+drCUI+"""' AND t.id = '"""+tarCUI+"""'
                DELETE r
                """)
        #j=j+1

    ################second part: load 

#    check3 = input("Finally kill again neo4j and Press return...")

    print('loading from the file to get link predictions.') 
    model = torch.load(embed+'/'+embed+str(f)+'/trained_model.pkl')
    tf=TriplesFactory.from_path_binary('dictionary.pt')

    print('Done')
    
    p=0
    ids_ground = {}
    rows=len(X_testPairs)
    cols = 3
    idList = [["","",""] for r in range(rows)]
    for j in X_testPairs.keys():
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


print("10-fold CV finished...calculating macro avgs:")
   
# report performance
print('Precision: %.3f (%.3f)' % (mean(precision), std(precision)))
print('recall: %.3f (%.3f)' % (mean(recall), std(recall)))
print('f1-score: %.3f (%.3f)' % (mean(f1), std(f1)))
