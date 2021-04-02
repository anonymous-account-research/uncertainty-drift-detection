import pandas as pd
from skmultiflow.drift_detection import PageHinkley, ADWIN
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import stats
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from WrapperClasses_Regression import MCDropoutREGWrapper
from WrapperClasses_Classification import MCDropoutCLFWrapper

def run_performance_experiment(stream, dataset, model_specification, detection, prediction_type, targets,
                               retrain=False, retrain_after=0, refit_use_Xtrain=True, features=1):
    dict_metrics = {}
    dict_metrics['Task'] = prediction_type
    dict_metrics['Dataset'] = dataset
    dict_metrics['Prediction_Algorithm'] = 'MC_Dropout'
    dict_metrics['Drift_Detection'] = detection.name
    dict_metrics['Retraining_After'] = retrain_after

    stream.restart()
    stream_length = stream.n_remaining_samples()
    train_size = int(stream_length * 0.05)
    X_train, y_train = stream.next_sample(train_size)

    model = MCDropoutCLFWrapper(model_specification, targets = targets, epochs=200, mcd_runs=50, debug=False)

    # fit algorithm
    model.fit(X_train, y_train)

    # skip adwin parameter validation data in stream
    _, _ = stream.next_sample(2 * train_size)


    #Test
    #X_test, y_test = stream.next_sample(2 * train_size)
    #y_pred, y_pred_uncertainty = model.predict(X_test)
    #predictions = np.argmax(y_pred, axis= 1)

    #print(accuracy_score(y_test, predictions))


    start = time.time()
    # start change detection algorithm
    metrics, detected_drifts, raw_results = detection.run_stream(stream, model, X_train, y_train, retrain,
                                                                 prediction_type, retrain_after, refit_use_Xtrain,
                                                                 features, targets)

    print(metrics)

    end = time.time()
    computation_time = end - start

    metrics['Computation_Time'] = computation_time

    return metrics, raw_results, detected_drifts



def get_model(features, targets):
    inputs = Input(shape=(features,))
    x_0 = Dense(32, activation='relu')(inputs)
    x_d0 = Dropout(0.1)(x_0, training=True)
    x_1 = Dense(16, activation='relu')(x_d0)
    x_d1 = Dropout(0.1)(x_1, training=True)
    x_2 = Dense(8, activation='relu')(x_d1)
    x_d2 = Dropout(0.1)(x_2, training=True)
    outputs = Dense(targets, activation='softmax')(x_d2)


    model_clf = Model(inputs, outputs)

    if targets > 2:
        model_clf.compile(optimizer='adam', loss='categorical_crossentropy')
    else:
        model_clf.compile(optimizer='adam', loss='binary_crossentropy')

    weights = model_clf.get_weights()
    model = [model_clf, weights]
    return model




def plot_performance_comparison(df_metrics):

    plt.plot(df_metrics.loc[df_metrics['Drift_Detection']=='adwin_error', 'Retraining_After'],df_metrics.loc[df_metrics['Drift_Detection']=='adwin_error', 'Performance_Metric1'], marker='o', label='adwin_error')
    plt.plot(df_metrics.loc[df_metrics['Drift_Detection']=='uninformed', 'Retraining_After'],df_metrics.loc[df_metrics['Drift_Detection']=='uninformed', 'Performance_Metric1'], marker='o', label='uninformed')
    plt.plot(df_metrics.loc[df_metrics['Drift_Detection']=='adwin_uncertainty', 'Retraining_After'],df_metrics.loc[df_metrics['Drift_Detection']=='adwin_uncertainty', 'Performance_Metric1'], marker='o',label='adwin_uncertainty')
    plt.plot(df_metrics.loc[df_metrics['Drift_Detection']=='ks_data', 'Retraining_After'],df_metrics.loc[df_metrics['Drift_Detection']=='ks_data', 'Performance_Metric1'], marker='o',label='ksw_data')

    plt.axhline(df_metrics.loc[df_metrics['Drift_Detection']=='no_retraining', 'Performance_Metric1'].values[0], ls='--', label='no_retraining')

    plt.xticks(np.arange(0, 41, 20))
    plt.xlabel('Retraining after X datapoints after drift detection')
    plt.ylabel('Performance metric1')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()
    
    
def print_evaluation(results):

    metrics = results['metrics']
    metrics = metrics.groupby('Detection').mean()

    print(metrics)
    
    
    # metrics = results['metrics']
    # errors = results['errors']
    #
    # columns = ['Drift_Detection', 'Retraining_After', 'Performance_Metric1','Performance_Metric2','Performance_Metric3', 'Performance_Metric4',
    #    'Detected_Drifts', 'Retraining_Counter', 'R_Pears', 'R_Spear']
    #
    # print(metrics.loc[metrics['Retraining_After']==0, columns ].groupby('Drift_Detection').mean())
    #
    # errors_uninformed_combined = errors['uninformed_1'] + errors['uninformed_2']+ errors['uninformed_3']+ errors['uninformed_4']+errors['uninformed_5']
    #
    # print("P-value adwin_uncertainty vs no_retraining:{}".format(stats.ttest_ind(errors['adwin_uncertainty_0'],        errors['no_retraining_0'])[1]))
    # print("P-value adwin_uncertainty vs uninformed: {}".format(stats.ttest_ind(errors['adwin_uncertainty_0'],        errors_uninformed_combined)[1]))
    # print("P-value adwin_uncertainty vs uninformed: {}".format(stats.ttest_ind(errors['adwin_uncertainty_0']*5,        errors_uninformed_combined)[1]))
    # print("P-value adwin_uncertainty vs ks_data: {}".format(stats.ttest_ind(errors['adwin_uncertainty_0'],        errors['ks_data_0'])[1]))
    # print("P-value adwin_uncertainty vs adwin_error: {}".format(stats.ttest_ind(errors['adwin_uncertainty_0'],        errors['adwin_error_0'])[1]))
    #
    #

    
def gridsearch_adwin_parameter(stream, algorithms, algorithm, drifts_to_detect, targets, features=1):
    
    stream.restart()
    
    five_percent = int(stream.n_remaining_samples() * 0.05)

    X_train, y_train = stream.next_sample(five_percent)
    
    #choose algorithm
    if algorithm in ['MCDropout', 'DeepEnsemble']:
        model = algorithms[algorithm](features, targets)
    else:
        model = algorithms[algorithm]()
    
    #fit algorithm
    model.fit(X_train, y_train)
    
    condition = False
    m = -1
    n = -1
    
    X_test, y_test = stream.next_sample(2 * five_percent)
    y_pred, y_pred_uncertainty = model.predict_grid(X_test)
   
    while condition == False: 
        print('-------------------------')
        param_box_upper = 10**m
        param_box_lower = 10**(n-1)

        #evaluate for param_box_upper    
        drifts_upper=[]
        adwin = ADWIN(param_box_upper)
                                                       
        for i in range(len(y_pred_uncertainty)):
            adwin.add_element(y_pred_uncertainty[i])
            if adwin.detected_change():
                # print('Change has been detected in data: - of index: ' + str(i))
                drifts_upper.append(i)
                adwin.reset()


        #evaluate for param_box_lower
        drifts_lower=[]
        adwin = ADWIN(param_box_lower)
                                                       
        for i in range(len(y_pred_uncertainty)):
            adwin.add_element(y_pred_uncertainty[i])
            if adwin.detected_change():
                # print('Change has been detected in data: - of index: ' + str(i))
                drifts_lower.append(i)
                adwin.reset()

                
        print('Adwin Parameter: {} - Detected Drifts: {}'.format(param_box_upper,len(drifts_upper)))
        print('Adwin Parameter: {} - Detected Drifts: {}'.format(param_box_lower,len(drifts_lower)))

        if len(drifts_upper) == drifts_to_detect:
            print('case1')
            parameter = param_box_upper
            condition = True
        
        elif len(drifts_lower) == drifts_to_detect:
            print('case1')
            parameter = param_box_lower
            condition = True
            
              
        elif (len(drifts_upper) > drifts_to_detect) & (len(drifts_lower) > drifts_to_detect):
            print('case2')
            m -= 1
            n -= 1
        
        elif (len(drifts_upper) < drifts_to_detect) & (len(drifts_lower) < drifts_to_detect) & (param_box_upper == 0.1):
            print('case3')

            parameter = 0.002
            condition = True

            
        elif (len(drifts_upper) > drifts_to_detect) & (len(drifts_lower) < drifts_to_detect):
            print('case4')

            m -= 0.05
            n += 0.05

    
    print(parameter)
    
    return parameter
    
def plot_drift_detection_summary(results_dict, stream_length):

    fig, ax = plt.subplots(1)

    ax.hlines(1,0, stream_length*0.85, color='tab:blue', label='ADWIN_Uncertainty')
    for i in results_dict['drifts']['adwin_uncertainty_0']:
        ax.plot(i,1, marker='x', color='red', markersize=8)

    ax.hlines(2,0,stream_length*0.85, color='tab:orange', label='KS_Data')
    for i in results_dict['drifts']['ks_data_0']:
        ax.plot(i,2, marker='x', color='red', markersize=8)

    ax.hlines(3,0,stream_length*0.85, color='tab:green', label='ADWIN_Error')
    for i in results_dict['drifts']['adwin_error_0']:
        ax.plot(i,3, marker='x', color='red', markersize=8)
        
    ax.hlines(4,0,stream_length*0.85, color='black', label='Uninformed_1')
    for i in results_dict['drifts']['uninformed_1']:
        ax.plot(i,4, marker='x', color='red', markersize=8)
    
    ax.hlines(5,0,stream_length*0.85, color='black', label='Uninformed_2')
    for i in results_dict['drifts']['uninformed_2']:
        ax.plot(i,5, marker='x', color='red', markersize=8)

    ax.hlines(6,0,stream_length*0.85, color='black', label='Uninformed_3')
    for i in results_dict['drifts']['uninformed_3']:
        ax.plot(i,6, marker='x', color='red', markersize=8)

    ax.hlines(7,0,stream_length*0.85, color='black', label='Uninformed_4')
    for i in results_dict['drifts']['uninformed_4']:
        ax.plot(i,7, marker='x', color='red', markersize=8)

    ax.hlines(8,0,stream_length*0.85, color='black', label='Uninformed_5')
    for i in results_dict['drifts']['uninformed_5']:
        ax.plot(i,8, marker='x', color='red', markersize=8)
        
    ax.set_yticks([])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('Drift Detection')
    
    plt.show()
    
def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))