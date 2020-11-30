import pandas as pd
import numpy as np
data = pd.read_excel('/Users/bobby/Desktop/ZGT/Work/dataset_hip_model2_v.08.xlsx',index_col='rec_id')

data = data[data.notnull().all(axis=1)]
data = data.reset_index(drop=True)

data

data_eng = {}
#Demographics
data_eng['gender'] = data['Geslacht']
data_eng['age'] = data['AGE']
#Cognitive Problems
data_eng['prone_to_delirium'] = data['kwetsbaar_op_delerium']
data_eng['memory_problem'] = data['geheugen_probl']
data_eng['previous_confusional_state'] = data['eerdere_verwardheid_bij_ziekte_of_opname']
#Activities of Daily Living
data_eng['katz_index_of_independence'] = data['Katz_adl_score']
#Assessment
data_eng['prone_to_under_nutrition'] = data['Kwetsbaar_op_ondervoeding']
data_eng['short_nutritional_assessment_score'] = data['SNAQ_core']
data_eng['unintentionally_lost_weight'] = data['onbedoeld_afgevallen']
data_eng['loss_of_appetite'] = data['Verminderde_eetlust']
data_eng['drink_or_tube_feeding'] = data['drink_of_sondevoeding']
data_eng['ASA_physical_status_classification'] = data['ASA2']
#Blood
data_eng['blood_pressure_systolic'] = data['NIBP_SYS']
data_eng['blood_pressure_diagstolic'] = data['NIBP_DIA']
#Cardiology
#data_eng['heart_rate'] = data['HF']
data_eng['width_of_QRS_complex_EKG'] = data['QRS']
data_eng['heart_axis_orientation_EKG'] = data['R_AS']
#Falling
data_eng['risk_of_falling'] = data['kwetsbaarheid_op_valrisico']
data_eng['a_fall_accident_in_past_6_months'] = data['gevallen_afgelopen_6mnd']
#Fracture
data_eng['fracture_type'] = data['soort_fractuur']
data_eng['laterality'] = data['aangedane_zijde']
data_eng['therapy_type'] = data['soort_therapie']
data_eng['care_paths_and_history'] = data['woonsit']
#Mobility
data_eng['pre_fracture_mobility'] = data['pre_fracture_mobility']
#Medication
data_eng['blood_thinner_medication'] = data['bloedverdunners']
data_eng['A02'] = data['A02']
data_eng['A10'] = data['A10']
data_eng['B01'] = data['B01']
data_eng['B02'] = data['B02']
data_eng['B03'] = data['B03']
data_eng['C01'] = data['C01']
data_eng['C03'] = data['C03']
data_eng['C07'] = data['C07']
data_eng['C08'] = data['C08']
data_eng['C09'] = data['C09']
data_eng['C10'] = data['C10']
data_eng['L04'] = data['L04']
data_eng['M01'] = data['M01']
data_eng['N05'] = data['N05']
data_eng['R03'] = data['R03']
#Specific Lab Test
data_eng['HB_ab'] = data['HB_afw']
data_eng['HT_ab'] = data['HT_afw']
data_eng['CRP_ab'] = data['CRP_afw']
data_eng['LEUC_ab'] = data['LEUC_afw']
data_eng['THR_ab'] = data['THR_afw']
#data_eng['BLGR'] = data['BLGR']
#data_eng['LRAI'] = data['LRAI']
data_eng['ALKF_ab'] = data['ALKF_afw']
data_eng['GGT_ab'] = data['GGT_afw']
data_eng['ASAT_ab'] = data['ASAT_afw']
data_eng['ALAT_ab'] = data['ALAT_afw']
data_eng['LDH1_ab'] = data['LDH1_afw']
data_eng['UREU_ab'] = data['UREU_afw']
data_eng['KREA_ab'] = data['KREA_afw']
data_eng['GFRM_ab'] = data['GFRM_afw']
data_eng['NA_ab'] = data['NA_afw']
data_eng['XKA_ab'] = data['XKA_afw']
data_eng['GLUCGLUC_ab'] = data['GLUCGLUC_afw']
#Radiology
data_eng['X-ray_finding_1'] = data['Xthorax_finding_infiltraat']
data_eng['X-ray_finding_2'] = data['Xthorax_finding_versterkte_tekening']
data_eng['X-ray_finding_3'] = data['Xthorax_finding_dc']
data_eng = pd.DataFrame(data=data_eng)

y = data['PP_overl30d']

replacements = {'gender':{'V':0,'M':1},
                'memory_problem':{'Nee':0,'Ja':1},
                'previous_confusional_state':{'Nee':0,'Ja':1},
                'unintentionally_lost_weight':{'nee':0,'ja, meer dan 6 kg in de laatste 6 maanden':1,
                                               'ja, meer dan 3 kg in de afgelopen maand':2},
                'loss_of_appetite':{'nee':0,'ja':1},
                'drink_or_tube_feeding':{'nee':0,'ja':1},
                'a_fall_accident_in_past_6_months':{'Ja':1,'Nee':0},
                'fracture_type':{'mediale collum fractuur gedisloceerd':0,'trochantere femur fractuur AO-A2':1,
                                 'trochantere femur fractuur AO-A3':2,'mediale collum fractuur niet gedisloceerd':3,
                                 'trochantere femur fractuur AO-A1':4,'subtrochantere femurfractuur':5,
                                 'unspecified':6,'Leeg':7},
                'laterality':{'links':0,'rechts':1},
                'therapy_type':{'intra medullaire pen heup (PFNA)':0,'hemiarthroplastiek heup (Kop-Hals Prothese)':1,
                                'glijdende heupschroef (DHS)':2,'overige':3,
                                'gecanuleerde schroef heup':4},
                'care_paths_and_history':{'zelfstandig':0,'zelfstandig met (dagelijkse/ADL) hulp':1,
                                          'verpleeghuis':2,'verzorgingshuis':3,
                                          'verpleeghuis revalidatie':4,'anders, nl.':5},
                'pre_fracture_mobility':{'mobiel zonder hulpmiddelen':0,'mobiel buiten met 2 hulpmiddelen of frame (bv. rollator)':1,
                                         'mobiel binnenshuis maar nooit naar buiten zonder hulp':2,'mobiel buiten met 1 hulpmiddel':3,
                                         'onbekend':4,'geen functionele mobiliteit (gebruikmakend van onderste extremiteit)':5},
                'blood_thinner_medication':{'nee, gebruik geen bloedverdunners':0,'ja, heb ze door gebruikt':1,'ja, gestopt vlgs afspraak':2}
                
               }
data_eng = data_eng.replace(replacements)

data_eng

X = []
for num in range(756):
    info = []
    for key in data_eng.keys():
        info.append(data_eng[key][num])
    X.append(info)
X = np.array(X)

np.save('data.npy',X)
np.save('label.npy',y)
