import numpy as np
X = np.load('data.npy')
y = np.load('label.npy')

name = ['gender', 'age', 'prone_to_delirium', 'memory_problem',
       'previous_confusional_state', 'katz_index_of_independence',
       'prone_to_under_nutrition', 'short_nutritional_assessment_score',
       'unintentionally_lost_weight', 'loss_of_appetite',
       'drink_or_tube_feeding', 'ASA_physical_status_classification',
       'blood_pressure_systolic', 'blood_pressure_diagstolic',
       'width_of_QRS_complex_EKG', 'heart_axis_orientation_EKG',
       'risk_of_falling', 'a_fall_accident_in_past_6_months', 'fracture_type',
       'laterality', 'therapy_type', 'care_paths_and_history',
       'pre_fracture_mobility', 'blood_thinner_medication', 'A02', 'A10',
       'B01', 'B02', 'B03', 'C01', 'C03', 'C07', 'C08', 'C09', 'C10', 'L04',
       'M01', 'N05', 'R03', 'HB_ab', 'HT_ab', 'CRP_ab', 'LEUC_ab', 'THR_ab',
       'ALKF_ab', 'GGT_ab', 'ASAT_ab', 'ALAT_ab', 'LDH1_ab', 'UREU_ab',
       'KREA_ab', 'GFRM_ab', 'NA_ab', 'XKA_ab', 'GLUCGLUC_ab',
       'X-ray_finding_1', 'X-ray_finding_2', 'X-ray_finding_3']

#########PearsonCorrelationCoefficient#############
corres = np.zeros(58)
for i in range(58):
    corres[i] = np.abs(np.corrcoef(X[:,i],y)[0,1])
top_k = 20
top_k_idx=corres.argsort()[::-1][0:top_k]
for i in range (20):
    print(name[top_k_idx[i]])

#########DistanceCorrelationCoefficient#############
from scipy.spatial.distance import correlation
corres = np.zeros(58)
for i in range(58):
    corres[i] = correlation(X[:,i],y)
top_k = 20
top_k_idx=corres.argsort()[::-1][0:top_k]
for i in range (20):
    print(name[top_k_idx[i]])
   
#########MaximalInformationCoefficient#############
from minepy import MINE
mine = MINE(alpha=0.6, c=15)
corres = np.zeros(58)
for i in range(58):
    mine.compute_score(X[:,i],y)
    corres[i] = mine.mic()
top_k = 20
top_k_idx=corres.argsort()[::-1][0:top_k]
for i in range (20):
    print(name[top_k_idx[i]])

#########RandomForest#############
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X,y)
top_k = 20
top_k_idx=rf.feature_importances_.argsort()[::-1][0:top_k]
for i in range (20):
    print(name[top_k_idx[i]])

X_new = []
X_new.append(X[:,0])
X_new.append(X[:,1])
X_new.append(X[:,3])
X_new.append(X[:,5])
X_new.append(X[:,11])
X_new.append(X[:,12])
X_new.append(X[:,13])
X_new.append(X[:,14])
X_new.append(X[:,15])
X_new.append(X[:,18])
X_new.append(X[:,20])
X_new.append(X[:,21])
X_new.append(X[:,22])
X_new.append(X[:,36])
X_new.append(X[:,40])
X_new.append(X[:,48])
X_new = np.array(X_new)
X_new = np.transpose(X_new)
X_new.shape

np.save('data_withFeatureSelection.npy',X_new)