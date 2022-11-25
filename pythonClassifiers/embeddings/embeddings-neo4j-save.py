# Define Neo4j connections
from neo4j import GraphDatabase
import pandas as pd
import numpy as np

host = 'bolt://localhost:7687'
user = 'neo4j'
password = 'iasis'
driver = GraphDatabase.driver(host,auth=(user, password))
                                         

def run_query(query, params={}):
    with driver.session() as session:
        result = session.run(query, params)
        return pd.DataFrame([r.values() for r in result], columns=result.keys())

#check1 = input("Press return to start:")
print('Program start. Run query to Neo4j..')

data = run_query("""
MATCH (s)-[r]->(t)
RETURN toString(id(s)) as source, toString(id(t)) AS target, type(r) as type
""")
#toString(id(s)) as source, toString(id(t)) AS target, type(r) as type

check2 = input("Done. Now kill neo4j (to allocate memory) and Press return to continue:")

from pykeen.triples import TriplesFactory

print('Insert neo4j graph into pykeen and save dictionary...')
###Insert neo4j graph into pykeen
tf = TriplesFactory.from_labeled_triples(
  data[["source", "type", "target"]].values,
  create_inverse_triples=False,
  entity_to_id=None,
  relation_to_id=None,
  compact_id=False,
  filter_out_candidate_inverse_relations=True,
  metadata=None,
)
tf.to_path_binary('dictionary.pt')
print('Done! ')

listEmbeddings = ['TransE', 'DistMult', 'RESCAL', 'HoLE']

print('Dividing pykeen data into training-test-validation.')
###Train  a specific embedding model (start from list[0]=TransE)
training, testing, validation = tf.split([.8, .1, .1])

print('Now preparing embeddings...')

from pykeen.pipeline import pipeline

for embed in listEmbeddings:
    print('training ', embed)
    result = pipeline(
        training=training,
        testing=testing,
        validation=validation,
        model=embed,
        stopper='early',
        epochs=2,
        dimensions=10,
        random_seed=42
    )
    print(embed, ': Saving created model to a local file...')
    result.save_to_directory(embed)
    break;

print('Done')
