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
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from util import run_performance_experiment, plot_performance_comparison, print_evaluation, gridsearch_adwin_parameter, \
    plot_drift_detection_summary, get_model
from WrapperClasses_Classification import MCDropoutCLFWrapper
from WrapperClasses_Regression import MCDropoutREGWrapper
from Detection_Strategies import Adwin_Uncertainty, Uninformed, Equal_Distribution, KS_Data, No_Retraining, Adwin_Error
import pickle

data = pd.read_csv('../01_Artificial_DataSets/friedman_with_2_abrupt_drift.csv')
dataset_name = 'friedman'
prediction_type = 'regression'
features = data.shape[1] - 1
stream_length = data.shape[0]
if prediction_type == 'regression':
    targets = 1
else:
    targets = data.iloc[:, -1].nunique()

print(data.head())

stream = DataStream(data)

one_percent = int(stream.n_remaining_samples() * 0.01)
two_percent = int(stream.n_remaining_samples() * 0.02)

# Define model
model = get_model(features, targets)

algorithm = 'MCDropout'

adwin_param = 0.0001

metrics_list = []
errors_dict = {}
drifts_dict = {}
raw_dict = {}

# Retraining after, here 0, Retraining as soon as drift is detected
ra = 0
adwin_uncertainty = Adwin_Uncertainty()
cd_strategy = adwin_uncertainty.name
print(adwin_uncertainty.name)

model_init = MCDropoutREGWrapper(model, epochs=1000, debug=False)
adwin_uncertainty.gridsearch_adwin_parameter(stream, model_init, 1, targets, features, starting_value=-2)

#adwin_param = 1e-2 #1e-4
#adwin_uncertainty.set_parameter(adwin_param)

metrics, raw_results, drift_points = run_performance_experiment(stream, dataset_name, model, adwin_uncertainty,
                                                                prediction_type, targets,
                                                                retrain=True, retrain_after=ra, refit_use_Xtrain=True,
                                                                adwin_parameter=adwin_param, features=features)

unc_series = pd.Series(raw_results['uncertainties']).rolling(window = 50).mean()
plt.plot(unc_series)
plt.show()

plt.plot(unc_series[6000:10000])
plt.show()

d1 = raw_results['uncertainties'][2500:5000].mean()
m1 = raw_results['uncertainties'][5000:7000].mean()
m2 = raw_results['uncertainties'][7000:9000].mean()
m3 = raw_results['uncertainties'][9000:11000].mean()

print('Before: ', m1)
print('During: ', m2)
print('After: ', m3)

print('Ratio: ', m2/m1)
print('Ratio Drift: ', d1/m1)


key = cd_strategy + '_' + str(ra)
errors_dict[key] = raw_results['errors']
drifts_dict[key] = drift_points
raw_dict[key] = raw_results
metrics['Alg_run'] = key
metrics['Detection'] = cd_strategy

metrics_list.append(metrics)
df_metrics_prelim = pd.DataFrame(metrics_list)

number_retrainings = metrics['retraining_counter']

# number_retrainings = 3





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
adwin_error = Adwin_Error()
adwin_error.gridsearch_adwin_parameter(stream, model_init, 1, targets, features, starting_value=-2)

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

#kswin_alpha = ks_data.gridsearch_kswin_parameter(stream, model_init, 3, targets, features, starting_value=-2)
kswin_alpha = 8.9e-07
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

print(ks_data.name)
cd_strategy = ks_data.name

ks_data.set_parameter(alpha = kswin_alpha)
metrics, raw_results, drift_points = run_performance_experiment(stream, dataset_name, model, ks_data,
                                                                prediction_type, targets,
                                                                retrain=True, retrain_after=ra,
                                                                refit_use_Xtrain=True, features=features)

key = cd_strategy + '_unlimited'
# errors_dict[key] = raw_results['errors']
errors_dict[key] = {}
drifts_dict[key] = drift_points
raw_dict[key] = raw_results
metrics['Alg_run'] = key
metrics['Detection'] = cd_strategy + '_unlimited'

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







#
# # # KSWIN, No Retraining, ADWIN Error
# no_retraining = No_Retraining()
# # ks_data = KS_Data(limit_drifts=False, detection_only=False, window_size=500, stat_size=200, retrain_after=50,
# #                   number_retrainings = number_retrainings) #500, 200, 100
# adwin_error = Adwin_Error()
#
# detection = {'no_retraining': no_retraining,
#              'ks_data': ks_data,
#              'adwin_error': adwin_error}
#
# for cd_strategy in ['ks_data']:  # , 'adwin_error']:
#
#     strategy = detection[cd_strategy]
#     print(cd_strategy)
#
#     kswin_param = strategy.gridsearch_kswin_parameter(stream, model_init, 1, targets, features)
#
#     metrics, raw_results, drift_points = run_performance_experiment(stream, dataset_name, model, strategy,
#                                                                     prediction_type, targets,
#                                                                     retrain=True, retrain_after=ra,
#                                                                     refit_use_Xtrain=True,
#                                                                     adwin_retrainings=number_retrainings,
#                                                                     adwin_parameter=adwin_param, features=features)
#
#     key = cd_strategy
#     # errors_dict[key] = raw_results['errors']
#     errors_dict[key] = {}
#     drifts_dict[key] = drift_points
#     raw_dict[key] = raw_results
#     metrics['Alg_run'] = key
#     metrics['Detection'] = cd_strategy
#
#     metrics_list.append(metrics)
#
# for cd_strategy in ['adwin_error']:
#     strategy = detection[cd_strategy]
#     print(cd_strategy)
#
#     strategy.gridsearch_adwin_parameter(stream, model_init, 1, targets, features)
#     # adwin_param = 1e-11
#
#     metrics, raw_results, drift_points = run_performance_experiment(stream, dataset_name, model, strategy,
#                                                                     prediction_type, targets,
#                                                                     retrain=True, retrain_after=ra,
#                                                                     refit_use_Xtrain=True,
#                                                                     adwin_retrainings=number_retrainings,
#                                                                     adwin_parameter=adwin_param, features=features)
#
#     key = cd_strategy
#     # errors_dict[key] = raw_results['errors']
#     errors_dict[key] = {}
#     drifts_dict[key] = drift_points
#     raw_dict[key] = raw_results
#     metrics['Alg_run'] = key
#     metrics['Detection'] = cd_strategy
#
#     metrics_list.append(metrics)
#
# df_metrics = pd.DataFrame(metrics_list)
#
# results_dict = {'metrics': df_metrics, 'errors': errors_dict, 'drifts': drifts_dict, 'raw': raw_dict}
# filename = 'results_' + algorithm + '_' + dataset_name + '.pickle'
# pickle.dump(results_dict, open(filename, 'wb'), protocol=4)
