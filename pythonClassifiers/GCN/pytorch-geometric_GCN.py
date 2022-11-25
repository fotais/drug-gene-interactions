import torch
import torch.nn.functional as F
import pandas as pd
from sentence_transformers import SentenceTransformer


rel_no=70

class SequenceEncoder(object):
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()


class TypesEncoder(object):
    def __init__(self, sep='|'):
        self.sep = sep

    def __call__(self, df):
        genres = set(g for col in df.values for g in col.split(self.sep))
        mapping = {genre: i for i, genre in enumerate(genres)}

        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            for genre in col.split(self.sep):
                x[i, mapping[genre]] = 1
        return x

class IdentityEncoder(object):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)

def normaliseRelation(rel):
    
    if rel=="ADMINISTERED_TO":
        norm_rel="ADMINISTERED_TO"
    elif rel=="ADMINISTERED_TO__SPEC__":
        norm_rel= "ADMINISTERED_TO"
    elif rel=="AFFECTS":
        norm_rel= "AFFECTS"        
    elif rel=="AFFECTS__SPEC__":
        norm_rel= "AFFECTS"
    elif rel=="ASSOCIATED_WITH":
        norm_rel= "ASSOCIATED_WITH"
    elif rel=="ASSOCIATED_WITH__INFER__":
        norm_rel= "ASSOCIATED_WITH"
    elif rel=="ASSOCIATED_WITH__SPEC__":
        norm_rel= "ASSOCIATED_WITH"
    elif rel=="AUGMENTS":
        norm_rel= "AUGMENTS"
    elif rel=="AUGMENTS__SPEC__":
        norm_rel= "AUGMENTS"
    elif rel=="CAUSES":
        norm_rel= "CAUSES"
    elif rel=="CAUSES__SPEC__":
        norm_rel= "CAUSES"
    elif rel=="COEXISTS_WITH":
        norm_rel= "COEXISTS_WITH"
    elif rel=="COEXISTS_WITH__SPEC__":
        norm_rel= "COEXISTS_WITH"
    elif rel=="compared_with":
        norm_rel= "compared_with"
    elif rel=="compared_with__SPEC__":
        norm_rel= "compared_with"
    elif rel=="COMPLICATES":
        norm_rel= "COMPLICATES"
    elif rel=="COMPLICATES__SPEC__":
        norm_rel= "COMPLICATES"
    elif rel=="CONVERTS_TO":
        norm_rel= "CONVERTS_TO"
    elif rel=="CONVERTS_TO__SPEC__":
        norm_rel= "CONVERTS_TO"
    elif rel=="DIAGNOSES":
        norm_rel= "DIAGNOSES"
    elif rel=="DIAGNOSES__SPEC__":
        norm_rel= "DIAGNOSES"
    elif rel=="different_from":
        norm_rel= "different_from"
    elif rel=="different_from__SPEC__":
        norm_rel= "different_from"
    elif rel=="different_than":
        norm_rel= "different_than"
    elif rel=="different_than__SPEC__":
        norm_rel= "different_than"
    elif rel=="DISRUPTS":
        norm_rel= "DISRUPTS"
    elif rel=="DISRUPTS__SPEC__":
        norm_rel= "DISRUPTS"
    elif rel=="higher_than":
        norm_rel= "higher_than"
    elif rel=="higher_than__SPEC__":
        norm_rel= "higher_than"
    elif rel=="INHIBITS":
        norm_rel= "INHIBITS"
    elif rel=="INHIBITS__SPEC__":
        norm_rel= "INHIBITS"
    elif rel=="INTERACTS_WITH":
        norm_rel= "INTERACTS_WITH"
    elif rel=="INTERACTS_WITH__INFER__":
        norm_rel= "INTERACTS_WITH"
    elif rel=="INTERACTS_WITH__SPEC__":
        norm_rel= "INTERACTS_WITH"
    elif rel=="IS_A":
        norm_rel= "IS_A"
    elif rel=="ISA":
        norm_rel= "ISA"
    elif rel=="LOCATION_OF":
        norm_rel= "LOCATION_OF"
    elif rel=="LOCATION_OF__SPEC__":
        norm_rel= "LOCATION_OF"
    elif rel=="lower_than":
        norm_rel= "lower_than"
    elif rel=="lower_than__SPEC__":
        norm_rel= "lower_than"
    elif rel=="MANIFESTATION_OF":
        norm_rel= "MANIFESTATION_OF"
    elif rel=="MANIFESTATION_OF__SPEC__":
        norm_rel= "MANIFESTATION_OF"
    elif rel=="METHOD_OF":
        norm_rel= "METHOD_OF"
    elif rel=="METHOD_OF__SPEC__":
        norm_rel= "METHOD_OF"
    elif rel=="OCCURS_IN":
        norm_rel= "OCCURS_IN"
    elif rel=="OCCURS_IN__SPEC__":
        norm_rel= "OCCURS_IN"
    elif rel=="PART_OF":
        norm_rel= "PART_OF"
    elif rel=="PART_OF__SPEC__":
        norm_rel= "PART_OF"
    elif rel=="PRECEDES":
        norm_rel= "PRECEDES"
    elif rel=="PRECEDES__SPEC__":
        norm_rel= "PRECEDES"
    elif rel=="PREDISPOSES":
        norm_rel= "PREDISPOSES"
    elif rel=="PREDISPOSES__SPEC__":
        norm_rel= "PREDISPOSES"
    elif rel=="PREVENTS":
        norm_rel= "PREVENTS"
    elif rel=="PREVENTS__SPEC__":
        norm_rel= "PREVENTS"
    elif rel=="PROCESS_OF":
        norm_rel= "PROCESS_OF"
    elif rel=="PROCESS_OF__SPEC__":
        norm_rel= "PROCESS_OF"
    elif rel=="PRODUCES":
        norm_rel= "PRODUCES"
    elif rel=="PRODUCES__SPEC__":
        norm_rel= "PRODUCES"
    elif rel=="same_as":
        norm_rel= "same_as"
    elif rel=="same_as__SPEC__":
        norm_rel= "same_as"
    elif rel=="STIMULATES":
        norm_rel= "STIMULATES"
    elif rel=="STIMULATES__SPEC__":
        norm_rel= "STIMULATES"
    elif rel=="TREATS":
        norm_rel= "TREATS"
    elif rel=="TREATS__INFER__":
        norm_rel= "TREATS"
    elif rel=="TREATS__SPEC__":
        norm_rel= "TREATS"
    elif rel=="USES":
        norm_rel= "USES"
    elif rel=="USES__SPEC__":
        norm_rel= "USES"
    elif rel=="MENTIONED_IN":
        norm_rel= "MENTIONED_IN"
    elif rel=="HAS_MESH":
        norm_rel= "HAS_MESH"
    else:
        norm_rel="ASSOCIATED_WITH"

    return norm_rel

def encodeEdgeTypes(df):
    reltypes = ["ADMINISTERED_TO","AFFECTS","ASSOCIATED_WITH","AUGMENTS","CAUSES","COEXISTS_WITH","compared_with","COMPLICATES","CONVERTS_TO","DIAGNOSES","different_from","different_than","DISRUPTS","higher_than","INHIBITS","INTERACTS_WITH","IS_A","ISA","LOCATION_OF","lower_than","MANIFESTATION_OF","METHOD_OF","OCCURS_IN","PART_OF","PRECEDES","PREDISPOSES","PREVENTS","PROCESS_OF","PRODUCES","same_as","STIMULATES","TREATS","USES","MENTIONED_IN","HAS_MESH"]
    mapping = {rtype: i for i, rtype in enumerate(reltypes)}
    x = torch.zeros(len(df), dtype=torch.int64)
    for i, col in enumerate(df.values):
        rel=normaliseRelation(col)
#        print('edgetype i, x[i]: ',i, type(mapping[rel]))
        x[i]= mapping[rel]
    return x



def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, sep='\t', **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping


def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, sep='\t', **kwargs)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
#        print('edge_attrs: ', edge_attrs)
        edge_attr = torch.cat(edge_attrs, dim=-1)
#        print('CONCATENATED edge_attr: ', edge_attr)
    return edge_index, edge_attr


################################################
###################Load CSV#####################
################################################

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#print(torch.version.cuda)

ent_file_path="/home/faisopos/workspace/neo4j-community-3.5.23/import/entities.tsv"
rel_file_path="/home/faisopos/workspace/neo4j-community-3.5.23/import/relations.tsv"
rel1_file_path="/home/faisopos/workspace/neo4j-community-3.5.23/import/relations_entities.tsv"
rel2_file_path="/home/faisopos/workspace/neo4j-community-3.5.23/import/relations_articles.tsv"
art_file_path="/home/faisopos/workspace/neo4j-community-3.5.23/import/articles.tsv"


entity_cui, entity_type = load_node_csv(
    ent_file_path, index_col='ID', encoders={
        'CUI': IdentityEncoder(dtype=torch.long),
        'SEM_TYPES': TypesEncoder()
    })


article_title, article_no = load_node_csv(
    art_file_path, index_col='AID', encoders={
        'TITLE': SequenceEncoder(),
        'ARTICLE_NO': IdentityEncoder(dtype=torch.int)
    })


from torch_geometric.data import HeteroData

data = HeteroData()

data['entity'].x = entity_cui
data['article'].x = article_title

#Insert and Encode relations

print('load entity-entity rels')
edge_index, edge_attr = load_edge_csv(
    rel1_file_path,
    src_index_col='NOD1',
    src_mapping=entity_type,
    dst_index_col='NOD2',
    dst_mapping=entity_type,
    encoders={'REFERENCES': IdentityEncoder(dtype=torch.long)#,  'RELATION': InteractsEncoder() 
},
)

data['entity', 'rel', 'entity'].edge_index = edge_index
data['entity', 'rel', 'entity'].edge_attr = edge_attr

print('load article-entity rels')

edge_index, edge_attr = load_edge_csv(
    rel2_file_path,
    src_index_col='NOD1',
    src_mapping=article_no,
    dst_index_col='NOD2',
    dst_mapping=entity_type,
    encoders={'REFERENCES': IdentityEncoder(dtype=torch.long)
#          'RELATION': InteractsEncoder()
    },
)

data['article', 'rel', 'entity'].edge_index = edge_index
data['article', 'rel', 'entity'].edge_attr = edge_attr

df = pd.read_csv(rel1_file_path, sep='\t')
df_rel=df["RELATION"]
edge_type = encodeEdgeTypes(df_rel)


################################################
###################GCN##########################
################################################


""""
Implements the link prediction task on the FB15k237 datasets according to the
`"Modeling Relational Data with Graph Convolutional Networks"
<https://arxiv.org/abs/1703.06103>`_ paper.
Caution: This script is executed in a full-batch fashion, and therefore needs
to run on CPU (following the experimental setup in the official paper).
"""

from torch.nn import Parameter
from tqdm import tqdm

from torch_geometric.datasets import RelLinkPredDataset
from torch_geometric.nn import GAE, RGCNConv


class RGCNEncoder(torch.nn.Module):
    def __init__(self, num_nodes, hidden_channels, num_relations):
        super().__init__()
        self.node_emb = Parameter(torch.Tensor(num_nodes, hidden_channels))
        self.conv1 = RGCNConv(hidden_channels, hidden_channels, num_relations,
                              num_blocks=5)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations,
                              num_blocks=5)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, edge_index, edge_type):
        x = self.node_emb
        x = self.conv1(x, edge_index, edge_type).relu_()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x


class DistMultDecoder(torch.nn.Module):
    def __init__(self, num_relations, hidden_channels):
        super().__init__()
        self.rel_emb = Parameter(torch.Tensor(num_relations, hidden_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.rel_emb)

    def forward(self, z, edge_index, edge_type):
        z_src, z_dst = z[edge_index[0]], z[edge_index[1]]
        rel = self.rel_emb[edge_type]
        return torch.sum(z_src * rel * z_dst, dim=1)


model = GAE(
    RGCNEncoder(data.num_nodes, hidden_channels=100,
                num_relations=rel_no),
    DistMultDecoder(rel_no // 2, hidden_channels=100),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def negative_sampling(edge_index, num_nodes):
    # Sample edges by corrupting either the subject or the object of each edge.
    mask_1 = torch.rand(edge_index.size(1)) < 0.5
    mask_2 = ~mask_1

    neg_edge_index = edge_index.clone()
    neg_edge_index[0, mask_1] = torch.randint(num_nodes, (mask_1.sum(), ))
    neg_edge_index[1, mask_2] = torch.randint(num_nodes, (mask_2.sum(), ))
    return neg_edge_index


def train():
    #print('Now training the RGCN model:')
    model.train()
    optimizer.zero_grad()

    #print('RGCNEncoder running...')  
    z = model.encode(data['entity', 'rel', 'entity'].edge_index, edge_type)
    #print('Decoder running...')
    pos_out = model.decode(z, train_edge_index, train_edge_type)

    #neg_edge_index = negative_sampling(train_edge_index, data.num_nodes)
    #neg_out = model.decode(z, neg_edge_index, train_edge_type)
    neg_out = model.decode(z, orig_neg_edge_index, orig_neg_edge_type)

    out = torch.cat([pos_out, neg_out])
    gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
    cross_entropy_loss = F.binary_cross_entropy_with_logits(out, gt)
    reg_loss = z.pow(2).mean() + model.decoder.rel_emb.pow(2).mean()
    loss = cross_entropy_loss + 1e-2 * reg_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()

    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    z = model.encode(data['entity', 'rel', 'entity'].edge_index, edge_type)

    #valid_mrr = compute_mrr(z, data.valid_edge_index, data.valid_edge_type)
    #test_mrr = compute_mrr(z, test_edge_index, test_edge_type)
    tps,fns = compute_poss(z, test_edge_index, test_edge_type, 1)
    tns,fps = compute_negs(z, test_neg_edge_index, test_neg_edge_type, 0)
    prec=tps/(tps+fps)
    rec=tps/(tps+fns)
    f1=2*prec*rec/(prec+rec)
    return prec, rec, f1 #valid_mrr, test_mrr



@torch.no_grad()
def compute_negs (z, eval_edge_index, eval_edge_type, pol):
    out = model.decode(z, eval_edge_index, eval_edge_type)
    #print('out object=', out)
    tns=0
    fps=0
    for predi in out:
        if predi>0.0:
            fps=fps+1
        else:
            tns=tns+1
    return tns,fps

@torch.no_grad()
def compute_poss (z, eval_edge_index, eval_edge_type, pol):
    out = model.decode(z, eval_edge_index, eval_edge_type)
    #print('out object=', out)
    tps=0
    fns=0
    for predi in out:
        if predi>0.0:
            tps=tps+1
        else:
            fns=fns+1
    return tps,fns



################################################
################### Groundtruth ################
################################################

groundtruth="/home/faisopos/workspace/python/GCN/posGroundtruth_corr.tsv"
neg_groundtruth="/home/faisopos/workspace/python/GCN/negGroundtruth_corr.tsv"

print('load entity-entity TRAIN rels')

ie = IdentityEncoder(dtype=torch.long)

df_ground= pd.read_csv(groundtruth, sep='\t')
df_neg_ground= pd.read_csv(neg_groundtruth, sep='\t')
neg_rtypes = df_neg_ground["RELATION"]
gener_refs = [0]*len(neg_rtypes)
df_neg_ground['REFERENCES'] = gener_refs
 
df = pd.concat([df_ground, df_neg_ground])

dfLabels=["NOD1","NOD2","RELATION"]
df_pairs_rtypes= df[dfLabels]
refs = df["REFERENCES"]

####undersampling original groundtruth ratio
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
#undersample = RandomUnderSampler(sampling_strategy=0.1)
#df_pairs_rtypes, refs = undersample.fit_resample(df_pairs_rtypes, refs)
###
print('Negative, Positive samples: ',Counter(refs))


################################################
################### k-Fold-CV ##################
################################################
from sklearn.model_selection import StratifiedKFold
k=10
i=0
VALIDATION_PRECISION = []
VALIDATION_RECALL= []
VALIDATION_FSCORE= []
skf = StratifiedKFold(n_splits = k, random_state = 1, shuffle = True)
for train_index, test_index in skf.split(df_pairs_rtypes, refs):
    i=i+1
    print('Fold ', i, ' - start training...')
    df_train=df.iloc[train_index]

    #undersampling training data, to set a different ratio
    undersample = RandomUnderSampler(sampling_strategy=0.05)
    df_train, df_train_refs = undersample.fit_resample(df_train, refs.iloc[train_index])
    print('Negative, Positive train samples: ',Counter(df_train_refs))

    df_test=df.iloc[test_index]
    #remove neg records from training...
    df_neg_train = df_train[df_train['REFERENCES'] == 0]
    df_train = df_train[df_train['REFERENCES'] > 0]
    #split pos/neg test records
    df_pos_test = df_test[df_test['REFERENCES'] > 0]
    df_neg_test = df_test[df_test['REFERENCES'] == 0] 

    t_train = df_train["RELATION"]
    pairLabels=["NOD1","NOD2"]
    p_train = df_train[pairLabels]
    a_train=df_train["REFERENCES"]
    src = [entity_type[index] for index in p_train["NOD1"]]
    dst = [entity_type[index] for index in p_train["NOD2"]]
    train_edge_index = torch.tensor([src, dst])
    train_edge_type = encodeEdgeTypes(t_train)
    train_edge_attrs = [ie(a_train)]
    train_edge_attr = torch.cat(train_edge_attrs, dim=-1)

####Keep Original negative sample...?
    neg_p_train = df_neg_train[pairLabels]
    nsrc = [entity_type[index] for index in neg_p_train["NOD1"]]
    ndst = [entity_type[index] for index in neg_p_train["NOD2"]]
    orig_neg_edge_index = torch.tensor([nsrc, ndst])
    neg_t_train = df_neg_train["RELATION"]
    orig_neg_edge_type = encodeEdgeTypes(neg_t_train)
####



##    p_train=pairs[train_index]
##    p_test=pairs[test_index]
##    t_train=rtypes[train_index]
##    t_test=rtypes[test_index]
##    a_train=refs[train_index]
##    a_test=refs[test_index]

    t_test = df_pos_test["RELATION"]
    p_test = df_pos_test[pairLabels]
    a_test=df_pos_test["REFERENCES"]
    src2 = [entity_type[index] for index in p_test["NOD1"]]
    dst2 = [entity_type[index] for index in p_test["NOD2"]]
    test_edge_index = torch.tensor([src2, dst2])
    test_edge_type = encodeEdgeTypes(t_test)
    test_edge_attrs = [ie(a_test)]
    test_edge_attr = torch.cat(test_edge_attrs, dim=-1)

    neg_t_test = df_neg_test["RELATION"]
    neg_p_test = df_neg_test[pairLabels]
    neg_a_test=df_neg_test["REFERENCES"]
    src3 = [entity_type[index] for index in neg_p_test["NOD1"]]
    dst3 = [entity_type[index] for index in neg_p_test["NOD2"]]
    test_neg_edge_index = torch.tensor([src3, dst3])
    test_neg_edge_type = encodeEdgeTypes(neg_t_test)
    ####

    prec=0.0
    rec=0.0
    f1=0.0
    ####Training....
    for epoch in range(1, 101):
        loss = train()
        if (epoch % 10) == 0:
            print(f'Epoch: {epoch:05d}, Loss: {loss:.4f}')
            prec, rec, f1 = test()
            print('Fold ', i, ', Epoch: ', epoch, '. Prec: ', prec, ' recall: ', rec, 'f1: ', f1)
    VALIDATION_PRECISION.append(prec)
    VALIDATION_RECALL.append(rec)
    VALIDATION_FSCORE.append(f1)
    
###################################################
##########Print marco-average of metric values#####
###################################################
                
print("\nTest Set Metrics of the trained model:")
print("Average Precision: ", mean(VALIDATION_PRECISION))
print("Average Recall: ", mean(VALIDATION_RECALL))
print("Average F1-score: ", mean(VALIDATION_FSCORE))
