from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=RuntimeWarning)

from scipy.io import arff

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skmultiflow.data import DataStream
from scipy import stats

from util import run_performance_experiment, plot_performance_comparison, print_evaluation, gridsearch_adwin_parameter, \
    plot_drift_detection_summary, get_model
from WrapperClasses_Classification import MCDropoutCLFWrapper
from WrapperClasses_Regression import MCDropoutREGWrapper
from Detection_Strategies import Adwin_Uncertainty, Uninformed, Equal_Distribution, KS_Data, No_Retraining, Adwin_Error
import pickle

data = arff.loadarff('INSECTS-incremental-abrupt_balanced_norm.arff')
data = pd.DataFrame(data[0])
data['class'] = data['class'].astype('category').cat.codes
dataset_name = 'INS-inc-abr-bal'
prediction_type = 'classification'
features = data.shape[1]-1
stream_length = data.shape[0]
if prediction_type == 'regression':
    targets = 1
else:
    targets = data.iloc[:,-1].nunique()

data.head()

stream = DataStream(data)

one_percent = int(stream.n_remaining_samples()*0.01)
two_percent = int(stream.n_remaining_samples()*0.02)


model = get_model(features, targets)
metrics_list = []
errors_dict = {}
drifts_dict = {}
raw_dict = {}
ra = 0



# ------------------
#
# ADWIN Uncertainty
#
# -----------------

adwin_uncertainty = Adwin_Uncertainty()

algorithm = 'MCDropout'

model_init = MCDropoutCLFWrapper(model, targets=targets, epochs=50, mcd_runs=50, debug=False)
#adwin_uncertainty.gridsearch_adwin_parameter(stream, model_init, 1, targets, features, starting_value=-1)
number_retrainings = 2

adwin_param = 0.1
adwin_uncertainty.set_parameter(adwin_param)

for ra in [0]:
    print(ra)
    print(adwin_uncertainty.name)
    cd_strategy = adwin_uncertainty.name
    metrics, raw_results, drift_points = run_performance_experiment(stream, dataset_name, model, adwin_uncertainty,
                                                                    prediction_type, targets,
                                                                    retrain=True, retrain_after=ra,
                                                                    refit_use_Xtrain=True, features=features)

    key = cd_strategy + ' ' + str(ra)
    errors_dict[key] = raw_results['errors']
    drifts_dict[key] = drift_points
    raw_dict[key] = raw_results
    metrics['Alg_run'] = key
    metrics['Detection'] = cd_strategy

    metrics_list.append(metrics)

df_metrics_prelim = pd.DataFrame(metrics_list)

number_retrainings = metrics['retraining_counter']
#number_retrainings = 2

# ------------------
#
# Uninformed
#
# -----------------


uninformed = Uninformed(number_retrainings)

for run in range(1,6):
    cd_strategy=uninformed.name
    print(cd_strategy)
    metrics, raw_results, drift_points = run_performance_experiment(stream, dataset_name, model, uninformed,
                                                                    prediction_type, targets,
                                                                    retrain=True, retrain_after=ra,
                                                                    refit_use_Xtrain=True, features=features)

    key = cd_strategy + '_' + str(run)
    errors_dict[key] = raw_results['errors']
    drifts_dict[key] = drift_points
    raw_dict[key] = raw_results
    metrics['Alg_run'] = key
    metrics['Detection'] = cd_strategy

    metrics_list.append(metrics)



# ------------------
#
# No Retraining + Equal distribution + ADWIN Error
#
# -----------------


no_retraining = No_Retraining()
equal_distribution= Equal_Distribution(number_retrainings)
adwin_error = Adwin_Error(prediction_type=prediction_type)

detection = {'no_retraining': no_retraining,
             'equal_distribution': equal_distribution,
             'adwin_error': adwin_error}

for cd_strategy in detection.keys():

    strategy = detection[cd_strategy]
    print(cd_strategy)

    metrics, raw_results, drift_points = run_performance_experiment(stream, dataset_name, model, strategy,
                                                                    prediction_type, targets,
                                                                    retrain=True, retrain_after=ra,
                                                                    refit_use_Xtrain=True, features=features)

    key = cd_strategy
    errors_dict[key] = raw_results['errors']
    drifts_dict[key] = drift_points
    raw_dict[key] = raw_results
    metrics['Alg_run'] = key
    metrics['Detection'] = cd_strategy

    metrics_list.append(metrics)




# ------------------
#
# KS Data
#
# -----------------

ks_data = KS_Data(number_retrainings)

print(ks_data.name)
cd_strategy = ks_data.name


#kswin_alpha = ks_data.gridsearch_kswin_parameter(stream, model_init, 1, targets, features, starting_value=-5)
kswin_alpha = 1e-10
ks_data.set_parameter(alpha = kswin_alpha)
metrics, raw_results, drift_points = run_performance_experiment(stream, dataset_name, model, ks_data,
                                                                prediction_type, targets,
                                                                retrain=True, retrain_after=ra,
                                                                refit_use_Xtrain=True, features=features)

key = cd_strategy
# errors_dict[key] = raw_results['errors']
errors_dict[key] = {}
drifts_dict[key] = drift_points
raw_dict[key] = raw_results
metrics['Alg_run'] = key
metrics['Detection'] = cd_strategy

metrics_list.append(metrics)


# ------------------
#
# KS Data, no limitation
#
# -----------------

ks_data = KS_Data(number_retrainings, limit_drifts=False)


cd_strategy = ks_data.name + '_unlimited'
print(cd_strategy)

ks_data.set_parameter(alpha = kswin_alpha)
metrics, raw_results, drift_points = run_performance_experiment(stream, dataset_name, model, ks_data,
                                                                prediction_type, targets,
                                                                retrain=True, retrain_after=ra,
                                                                refit_use_Xtrain=True, features=features)

key = cd_strategy
# errors_dict[key] = raw_results['errors']
errors_dict[key] = {}
drifts_dict[key] = drift_points
raw_dict[key] = raw_results
metrics['Alg_run'] = key
metrics['Detection'] = cd_strategy

metrics_list.append(metrics)



# ------------------
#
# Print final results
#
# -----------------


df_metrics = pd.DataFrame(metrics_list)

results_dict = {'metrics': df_metrics, 'errors': errors_dict, 'drifts': drifts_dict, 'raw': raw_dict}

#plot_performance_comparison(df_metrics)

#Save data
filename = 'results_' + algorithm + '_' + dataset_name +'.pickle'
pickle.dump(results_dict, open(filename, 'wb'), protocol=4)

print_evaluation(results_dict)
