import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler

data = pd.read_csv("DTI-enriched_sematyp_featuresExtended_corrected.csv")

#train model

feature_cols=["nod1_aapp", "nod1_acab", "nod1_acty", "nod1_aggp", "nod1_amas", "nod1_amph", "nod1_anab", "nod1_anim", "nod1_anst", "nod1_antb", "nod1_arch", "nod1_bacs", "nod1_bact", "nod1_bdsu", "nod1_bdsy", "nod1_bhvr", "nod1_biof", "nod1_bird", "nod1_blor", "nod1_bmod", "nod1_bodm", "nod1_bpoc", "nod1_bsoj", "nod1_celc", "nod1_celf", "nod1_cell", "nod1_cgab", "nod1_chem", "nod1_chvf", "nod1_chvs", "nod1_clas", "nod1_clna", "nod1_clnd", "nod1_cnce", "nod1_comd", "nod1_crbs", "nod1_diap", "nod1_dora", "nod1_drdd", "nod1_dsyn", "nod1_edac", "nod1_eehu", "nod1_elii", "nod1_emod", "nod1_emst", "nod1_enty", "nod1_enzy", "nod1_euka", "nod1_evnt", "nod1_famg", "nod1_ffas", "nod1_fish", "nod1_fndg", "nod1_fngs", "nod1_food", "nod1_ftcn", "nod1_genf", "nod1_geoa", "nod1_gngm", "nod1_gora", "nod1_grpa", "nod1_grup", "nod1_hcpp", "nod1_hcro", "nod1_hlca", "nod1_hops", "nod1_horm", "nod1_humn", "nod1_idcn", "nod1_imft", "nod1_inbe", "nod1_inch", "nod1_inpo", "nod1_inpr", "nod1_irda", "nod1_lang", "nod1_lbpr", "nod1_lbtr", "nod1_mamm", "nod1_mbrt", "nod1_mcha", "nod1_medd", "nod1_menp", "nod1_mnob", "nod1_mobd", "nod1_moft", "nod1_mosq", "nod1_neop", "nod1_nnon", "nod1_npop", "nod1_nusq", "nod1_ocac", "nod1_ocdi", "nod1_orch", "nod1_orga", "nod1_orgf", "nod1_orgm", "nod1_orgt", "nod1_ortf", "nod1_patf", "nod1_phob", "nod1_phpr", "nod1_phsf", "nod1_phsu", "nod1_plnt", "nod1_podg", "nod1_popg", "nod1_prog", "nod1_pros", "nod1_qlco", "nod1_qnco", "nod1_rcpt", "nod1_rept", "nod1_resa", "nod1_resd", "nod1_rnlw", "nod1_sbst", "nod1_shro", "nod1_socb", "nod1_sosy", "nod1_spco", "nod1_tisu", "nod1_tmco", "nod1_topp", "nod1_virs", "nod1_vita", "nod1_vtbt", "rel1_ADMINISTERED_TO", "rel1_AFFECTS", "rel1_ASSOCIATED_WITH", "rel1_AUGMENTS", "rel1_CAUSES", "rel1_COEXISTS_WITH", "rel1_compared_with", "rel1_COMPLICATES", "rel1_CONVERTS_TO", "rel1_DIAGNOSES", "rel1_different_from", "rel1_different_than", "rel1_DISRUPTS", "rel1_higher_than", "rel1_INHIBITS", "rel1_INTERACTS_WITH", "rel1_IS_A", "rel1_ISA", "rel1_LOCATION_OF", "rel1_lower_than", "rel1_MANIFESTATION_OF", "rel1_METHOD_OF", "rel1_OCCURS_IN", "rel1_PART_OF", "rel1_PRECEDES", "rel1_PREDISPOSES", "rel1_PREVENTS", "rel1_PROCESS_OF", "rel1_PRODUCES", "rel1_same_as", "rel1_STIMULATES", "rel1_TREATS", "rel1_USES", "nod2_aapp", "nod2_acab", "nod2_acty", "nod2_aggp", "nod2_amas", "nod2_amph", "nod2_anab", "nod2_anim", "nod2_anst", "nod2_antb", "nod2_arch", "nod2_bacs", "nod2_bact", "nod2_bdsu", "nod2_bdsy", "nod2_bhvr", "nod2_biof", "nod2_bird", "nod2_blor", "nod2_bmod", "nod2_bodm", "nod2_bpoc", "nod2_bsoj", "nod2_celc", "nod2_celf", "nod2_cell", "nod2_cgab", "nod2_chem", "nod2_chvf", "nod2_chvs", "nod2_clas", "nod2_clna", "nod2_clnd", "nod2_cnce", "nod2_comd", "nod2_crbs", "nod2_diap", "nod2_dora", "nod2_drdd", "nod2_dsyn", "nod2_edac", "nod2_eehu", "nod2_elii", "nod2_emod", "nod2_emst", "nod2_enty", "nod2_enzy", "nod2_euka", "nod2_evnt", "nod2_famg", "nod2_ffas", "nod2_fish", "nod2_fndg", "nod2_fngs", "nod2_food", "nod2_ftcn", "nod2_genf", "nod2_geoa", "nod2_gngm", "nod2_gora", "nod2_grpa", "nod2_grup", "nod2_hcpp", "nod2_hcro", "nod2_hlca", "nod2_hops", "nod2_horm", "nod2_humn", "nod2_idcn", "nod2_imft", "nod2_inbe", "nod2_inch", "nod2_inpo", "nod2_inpr", "nod2_irda", "nod2_lang", "nod2_lbpr", "nod2_lbtr", "nod2_mamm", "nod2_mbrt", "nod2_mcha", "nod2_medd", "nod2_menp", "nod2_mnob", "nod2_mobd", "nod2_moft", "nod2_mosq", "nod2_neop", "nod2_nnon", "nod2_npop", "nod2_nusq", "nod2_ocac", "nod2_ocdi", "nod2_orch", "nod2_orga", "nod2_orgf", "nod2_orgm", "nod2_orgt", "nod2_ortf", "nod2_patf", "nod2_phob", "nod2_phpr", "nod2_phsf", "nod2_phsu", "nod2_plnt", "nod2_podg", "nod2_popg", "nod2_prog", "nod2_pros", "nod2_qlco", "nod2_qnco", "nod2_rcpt", "nod2_rept", "nod2_resa", "nod2_resd", "nod2_rnlw", "nod2_sbst", "nod2_shro", "nod2_socb", "nod2_sosy", "nod2_spco", "nod2_tisu", "nod2_tmco", "nod2_topp", "nod2_virs", "nod2_vita", "nod2_vtbt", "rel2_ADMINISTERED_TO", "rel2_AFFECTS", "rel2_ASSOCIATED_WITH", "rel2_AUGMENTS", "rel2_CAUSES", "rel2_COEXISTS_WITH", "rel2_compared_with", "rel2_COMPLICATES", "rel2_CONVERTS_TO", "rel2_DIAGNOSES", "rel2_different_from", "rel2_different_than", "rel2_DISRUPTS", "rel2_higher_than", "rel2_INHIBITS", "rel2_INTERACTS_WITH", "rel2_IS_A", "rel2_ISA", "rel2_LOCATION_OF", "rel2_lower_than", "rel2_MANIFESTATION_OF", "rel2_METHOD_OF", "rel2_OCCURS_IN", "rel2_PART_OF", "rel2_PRECEDES", "rel2_PREDISPOSES", "rel2_PREVENTS", "rel2_PROCESS_OF", "rel2_PRODUCES", "rel2_same_as", "rel2_STIMULATES", "rel2_TREATS", "rel2_USES", "nod3_aapp", "nod3_acab", "nod3_acty", "nod3_aggp", "nod3_amas", "nod3_amph", "nod3_anab", "nod3_anim", "nod3_anst", "nod3_antb", "nod3_arch", "nod3_bacs", "nod3_bact", "nod3_bdsu", "nod3_bdsy", "nod3_bhvr", "nod3_biof", "nod3_bird", "nod3_blor", "nod3_bmod", "nod3_bodm", "nod3_bpoc", "nod3_bsoj", "nod3_celc", "nod3_celf", "nod3_cell", "nod3_cgab", "nod3_chem", "nod3_chvf", "nod3_chvs", "nod3_clas", "nod3_clna", "nod3_clnd", "nod3_cnce", "nod3_comd", "nod3_crbs", "nod3_diap", "nod3_dora", "nod3_drdd", "nod3_dsyn", "nod3_edac", "nod3_eehu", "nod3_elii", "nod3_emod", "nod3_emst", "nod3_enty", "nod3_enzy", "nod3_euka", "nod3_evnt", "nod3_famg", "nod3_ffas", "nod3_fish", "nod3_fndg", "nod3_fngs", "nod3_food", "nod3_ftcn", "nod3_genf", "nod3_geoa", "nod3_gngm", "nod3_gora", "nod3_grpa", "nod3_grup", "nod3_hcpp", "nod3_hcro", "nod3_hlca", "nod3_hops", "nod3_horm", "nod3_humn", "nod3_idcn", "nod3_imft", "nod3_inbe", "nod3_inch", "nod3_inpo", "nod3_inpr", "nod3_irda", "nod3_lang", "nod3_lbpr", "nod3_lbtr", "nod3_mamm", "nod3_mbrt", "nod3_mcha", "nod3_medd", "nod3_menp", "nod3_mnob", "nod3_mobd", "nod3_moft", "nod3_mosq", "nod3_neop", "nod3_nnon", "nod3_npop", "nod3_nusq", "nod3_ocac", "nod3_ocdi", "nod3_orch", "nod3_orga", "nod3_orgf", "nod3_orgm", "nod3_orgt", "nod3_ortf", "nod3_patf", "nod3_phob", "nod3_phpr", "nod3_phsf", "nod3_phsu", "nod3_plnt", "nod3_podg", "nod3_popg", "nod3_prog", "nod3_pros", "nod3_qlco", "nod3_qnco", "nod3_rcpt", "nod3_rept", "nod3_resa", "nod3_resd", "nod3_rnlw", "nod3_sbst", "nod3_shro", "nod3_socb", "nod3_sosy", "nod3_spco", "nod3_tisu", "nod3_tmco", "nod3_topp", "nod3_virs", "nod3_vita", "nod3_vtbt", "rel3_ADMINISTERED_TO", "rel3_AFFECTS", "rel3_ASSOCIATED_WITH", "rel3_AUGMENTS", "rel3_CAUSES", "rel3_COEXISTS_WITH", "rel3_compared_with", "rel3_COMPLICATES", "rel3_CONVERTS_TO", "rel3_DIAGNOSES", "rel3_different_from", "rel3_different_than", "rel3_DISRUPTS", "rel3_higher_than", "rel3_INHIBITS", "rel3_INTERACTS_WITH", "rel3_IS_A", "rel3_ISA", "rel3_LOCATION_OF", "rel3_lower_than", "rel3_MANIFESTATION_OF", "rel3_METHOD_OF", "rel3_OCCURS_IN", "rel3_PART_OF", "rel3_PRECEDES", "rel3_PREDISPOSES", "rel3_PREVENTS", "rel3_PROCESS_OF", "rel3_PRODUCES", "rel3_same_as", "rel3_STIMULATES", "rel3_TREATS", "rel3_USES"]

X=data[feature_cols]
y=data["GROUNDTRUTH"]

#undersampling
undersample = RandomUnderSampler(sampling_strategy='majority')
X, y = undersample.fit_resample(X, y)
print('Negative, Positive samples: ',Counter(y))
   
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
# prepare the cross-validation procedure
print('prepare the cross-validation procedure...')
cv = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
# create model
print('create model...')
model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=20000)
# evaluate model
print('evaluate model...')

precision = np.zeros(10)
recall = np.zeros(10)
f1=np.zeros(10)
i=0

from sklearn import metrics

for train_index, test_index in cv.split(X, y):
   X_train, X_test = X.iloc[train_index], X.iloc[test_index] 
   y_train, y_test = y.iloc[train_index], y.iloc[test_index]

   model.fit(X_train, y_train)
   print("Fold ", i+1, ", prediction")
   y_pred = model.predict(X_test)

   precision[i]=metrics.precision_score(y_test, y_pred)
   recall[i]=metrics.recall_score(y_test, y_pred)
   f1[i]=metrics.f1_score(y_test, y_pred)
   print("f1[i]=",f1[i])
   i=i+1

# report performance
print('Precision: %.3f (%.3f)' % (mean(precision), std(precision)))
print('recall: %.3f (%.3f)' % (mean(recall), std(recall)))
print('f1-score: %.3f (%.3f)' % (mean(f1), std(f1)))

