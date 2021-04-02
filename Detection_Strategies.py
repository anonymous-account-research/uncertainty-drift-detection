import pandas as pd
from skmultiflow.drift_detection import PageHinkley, ADWIN
from sklearn.metrics import log_loss, mean_squared_error, accuracy_score, mean_absolute_error, f1_score, matthews_corrcoef, mean_squared_error, mean_squared_log_error, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from skmultiflow.data import DataStream
from scipy import stats
import time
import random
from KSWIN import KSWIN
from util import smape
import copy
from keras.utils import to_categorical
from scipy.stats import spearmanr
from sklearn.preprocessing import LabelBinarizer
import time


def compute_log_loss_per_sample(y_test_cat, y_pred_proba, eps = 1e-5, labels = None):
    y_pred_proba = y_pred_proba.astype(np.float64)
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    lb = LabelBinarizer()

    if labels is not None:
        lb.fit(labels)
    else:
        lb.fit(y_test_cat)

    transformed_labels = lb.transform(y_test_cat)

    y_pred_proba /= y_pred_proba.sum(axis=1)[:, np.newaxis]
    log_loss_per_sample = -(transformed_labels * np.log(y_pred_proba)).sum(axis=1)
    return log_loss_per_sample


def compute_metrics(prediction_type, true_values, predictions, class_probabilities, targets, uncertainties,
                    detected_drifts, retraining_counter):
    metrics = {}

    if prediction_type == 'regression':
        metrics['type'] = 'regression'

        metrics['MAE'] = mean_absolute_error(true_values, predictions)
        metrics['MSE'] = mean_squared_error(true_values, predictions)
        metrics['SMAPE'] = smape(true_values, predictions)
        errors = pd.Series(np.square(np.absolute(true_values - predictions)))

    if prediction_type == 'classification':

        metrics['type'] = 'classification'

        metrics['Acc'] = accuracy_score(true_values, predictions)

        if (class_probabilities.shape[1] != np.unique(true_values).shape[0]):
            metrics['Log-loss'] = log_loss(true_values, class_probabilities, labels = np.arange(targets))
        else:
            metrics['Log-loss'] = log_loss(true_values, class_probabilities)

        metrics['MCC'] = matthews_corrcoef(true_values, predictions)

        if targets > 2:
            metrics['AUC'] = roc_auc_score(true_values, class_probabilities, average='weighted', multi_class='ovo',
                                           labels=np.arange(targets))
        else:
            metrics['AUC'] = roc_auc_score(true_values, class_probabilities[:, 1])

        errors = pd.Series((~np.equal(true_values, predictions)).astype(int))

    # y_true_cat = to_categorical(true_values, num_classes=targets)
    # log_loss_per_sample = compute_log_loss_per_sample(y_true_cat, class_probabilities)
    metrics['r_spear'] = spearmanr(errors, uncertainties)[0]
    metrics['r_pears'] = np.corrcoef(errors, uncertainties)[0, 1]

    metrics['detected_drift_numbers'] = len(detected_drifts)
    metrics['retraining_counter'] = retraining_counter

    return metrics, errors



def execute_stream_no_drift_detection(stream, model, retrain_points, refit_use_Xtrain, X_train, y_train, retrain_size,
                                      prediction_type = 'regression'):

    predictions = []
    true_values = []
    uncertainties = []
    class_probabilities = []
    X_retrain = []

    overall_i = 0
    retraining_counter = 0

    while stream.n_remaining_samples() > 1:
        # Standard batch size for computation
        batch_size = 1000

        # Check if remaining stream has still enough instances
        if (stream.n_remaining_samples() < batch_size):
            batch_size = stream.n_remaining_samples()

        X_test, y_test = stream.next_sample(batch_size)
        y_pred, y_pred_uncertainty = model.predict(X_test)

        if prediction_type == 'regression':
            predictions.append(y_pred[0])
        if prediction_type == 'classification':
            predictions.append(y_pred)
        true_values.append(y_test)
        uncertainties.append(y_pred_uncertainty)
        # class_probabilities.append(y_probabilities)
        X_retrain.append(X_test)

        for j in range(len(y_pred_uncertainty)):

            if (overall_i + j) in retrain_points:
                retraining_counter += 1
                if refit_use_Xtrain:

                    # Remove training point
                    _ = retrain_points.pop(0)

                    # Remove excess training data
                    _ = X_retrain.pop()
                    _ = true_values.pop()
                    _ = predictions.pop()
                    _ = uncertainties.pop()
                    # _ = class_probabilities.pop()

                    # Add data up to drift instead
                    X_retrain.append(X_test[:j, :])
                    true_values.append(y_test[:j])
                    if prediction_type == 'regression':
                        predictions.append(y_pred[0][:j])
                    if prediction_type == 'classification':
                        predictions.append(y_pred[:j, :])
                    uncertainties.append(y_pred_uncertainty[:j])
                    # class_probabilities.append(y_probabilities[:j])

                    # Stack training data
                    X_retrain_array = np.vstack(X_retrain)
                    true_values_array = np.hstack(true_values)

                    # Attention, need to adapt X_retrain
                    model.refit(np.concatenate([X_train, X_retrain_array[-retrain_size:, :]]),
                                np.concatenate([y_train, true_values_array[-retrain_size:]]))
                    batch_size = j

                    # Restart stream and set to correct position
                    stream.restart()
                    _, _ = stream.next_sample((3 * X_train.shape[0]) + (overall_i + j))
                    break
                else:
                    model.refit(np.concatenate(X_retrain[-retrain_size:]), np.concatenate(true_values[-retrain_size:]))

        # Need to adapt overall_i
        overall_i += batch_size

        results = {}
        results['predictions'] = predictions
        results['true_values'] = true_values
        results['uncertainties'] = uncertainties
        results['class_probabilities'] = class_probabilities
        results['X_retrain'] = X_retrain
        results['retraining_counter'] = retraining_counter

    return results


class Adwin_Uncertainty():
    def __init__(self):
        self.name = 'adwin_uncertainty'
        self.delta = 0.002

    def set_parameter(self, delta):
        self.delta = delta

    def gridsearch_adwin_parameter(self, stream, model, drifts_to_detect, targets, features=1, starting_value = -3):

        stream.restart()

        five_percent = int(stream.n_remaining_samples() * 0.05)

        X_train, y_train = stream.next_sample(five_percent)

        # fit algorithm
        model.fit(X_train, y_train)

        condition = False
        m = starting_value #-1
        n = starting_value #-1

        X_test, y_test = stream.next_sample(2 * five_percent)
        y_pred, y_pred_uncertainty = model.predict(X_test)

        # series_uncertainty = pd.Series(y_pred_uncertainty)
        # plt.plot(series_uncertainty.rolling(window=20).mean())
        # plt.show()

        while condition == False:
            print('-------------------------')
            param_box_upper = 10 ** m
            param_box_lower = 10 ** (n - 1)

            # evaluate for param_box_upper
            drifts_upper = []
            adwin = ADWIN(param_box_upper)

            for i in range(len(y_pred_uncertainty)):
                adwin.add_element(y_pred_uncertainty[i])
                if adwin.detected_change():
                    # print('Change has been detected in data: - of index: ' + str(i))
                    drifts_upper.append(i)
                    adwin.reset()

            # evaluate for param_box_lower
            drifts_lower = []
            adwin = ADWIN(param_box_lower)

            for i in range(len(y_pred_uncertainty)):
                adwin.add_element(y_pred_uncertainty[i])
                if adwin.detected_change():
                    # print('Change has been detected in data: - of index: ' + str(i))
                    drifts_lower.append(i)
                    adwin.reset()

            print('Adwin Parameter: {} - Detected Drifts: {}'.format(param_box_upper, len(drifts_upper)))
            print('Adwin Parameter: {} - Detected Drifts: {}'.format(param_box_lower, len(drifts_lower)))

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

            elif (len(drifts_upper) < drifts_to_detect) & (len(drifts_lower) < drifts_to_detect) & (
                    param_box_upper == 0.1):
                print('case3')

                parameter = 0.002
                condition = True


            elif (len(drifts_upper) > drifts_to_detect) & (len(drifts_lower) < drifts_to_detect):
                print('case4')

                m -= 0.05
                n += 0.05

        print('Final ADWIN paramter: ', parameter)
        self.delta = parameter



    def run_stream(self, stream, model, X_train, y_train, retrain, prediction_type, retrain_after, refit_use_Xtrain,
                   ks_features, targets):

        adwin = ADWIN(self.delta)

        retrain_size = int((stream.n_remaining_samples()/0.85)*0.01)
        train_size = X_train.shape[0]
        #print(retrain_size)

        predictions = []
        true_values = []
        uncertainties = []
        class_probabilities = []
        accepted_drifts = []
        detected_drifts = []
        X_retrain = []

        overall_i = 0
        next_retrain = -1
        retraining_counter = 0

        while stream.n_remaining_samples() > 1:

            # Standard batch size for computation
            batch_size = 1000

            # Check if remaining stream has still enough instances
            if (stream.n_remaining_samples() < batch_size):
                batch_size = stream.n_remaining_samples()

            X_test, y_test = stream.next_sample(batch_size)
            y_pred, y_pred_uncertainty = model.predict(X_test)

            if prediction_type == 'regression':
                predictions.append(y_pred[0])
            if prediction_type == 'classification':
                predictions.append(y_pred)

            true_values.append(y_test)
            uncertainties.append(y_pred_uncertainty)
            #class_probabilities.append(y_probabilities)
            X_retrain.append(X_test)

            for j in range(len(y_pred_uncertainty)):

                adwin.add_element(y_pred_uncertainty[j])
                if adwin.detected_change():
                    #print('Drift {}'.format(j))
                    if not accepted_drifts:
                        if retrain==True:
                            next_retrain = overall_i + j + retrain_after
                    elif (overall_i + j)-accepted_drifts[-1] > retrain_after:
                        if retrain==True:
                            next_retrain = overall_i + j + retrain_after

                    if not accepted_drifts:
                        accepted_drifts.append(overall_i + j)
                    elif (overall_i + j)-accepted_drifts[-1]> retrain_after:
                        accepted_drifts.append(overall_i + j)

                    detected_drifts.append(overall_i + j)
                    adwin.reset()

                if (overall_i + j) == next_retrain:

                    retraining_counter += 1
                    if refit_use_Xtrain:

                        # Remove excess training data
                        _ = X_retrain.pop()
                        _ = true_values.pop()
                        _ = predictions.pop()
                        _ = uncertainties.pop()
                        #_ = class_probabilities.pop()

                        # Add data up to drift instead
                        X_retrain.append(X_test[:j, :])
                        true_values.append(y_test[:j])
                        if prediction_type == 'regression':
                            predictions.append(y_pred[0][:j])
                        if prediction_type == 'classification':
                            predictions.append(y_pred[:j,:])
                        uncertainties.append(y_pred_uncertainty[:j])
                        #class_probabilities.append(y_probabilities[:j])


                        # Stack training data
                        X_retrain_array = np.vstack(X_retrain)
                        true_values_array = np.hstack(true_values)

                        # Attention, need to adapt X_retrain
                        model.refit(np.concatenate([X_train, X_retrain_array[-retrain_size:, :]]),
                                    np.concatenate([y_train, true_values_array[-retrain_size:]]))
                        batch_size = j
                        next_retrain = -1

                        # Restart stream and set to correct position
                        stream.restart()
                        _, _ = stream.next_sample((3 * X_train.shape[0]) + (overall_i + j))
                        break
                    else:
                        model.refit(np.concatenate(X_retrain[-200:]), np.concatenate(true_values[-200:]))

            # Need to adapt overall_i
            overall_i += batch_size

            # if (i % 100) == 0:
            #     print('100 instances processed')
            #     print('-----------------------')

        if prediction_type == 'regression':
            predictions = np.hstack(predictions)
        if prediction_type == 'classification':
            predictions = np.vstack(predictions)
            class_probabilities = predictions
            predictions = np.argmax(predictions, axis=1)

        true_values = np.hstack(true_values)
        #class_probabilities = np.vstack(class_probabilities)
        uncertainties = pd.Series(np.hstack(uncertainties))

        metrics, errors = compute_metrics(prediction_type, true_values, predictions, class_probabilities, targets,
                                          uncertainties, detected_drifts, retraining_counter)

        print('ADWIN Uncertainty: ', detected_drifts)



        raw_results = {'uncertainties': uncertainties, 'tv': true_values, 'tv_cat': None, 'preds': predictions, 'probs': class_probabilities, 'errors': errors}


        return metrics, detected_drifts, raw_results

class Uninformed():
    def __init__(self, number_retrainings):
        self.name = 'uninformed'
        self.number_retrainings = number_retrainings

    def run_stream(self, stream, model,  X_train, y_train, retrain, prediction_type, retrain_after, refit_use_Xtrain,
                   ks_features, targets):

        retrain_size = int((stream.n_remaining_samples()/0.850)*0.01)
        train_size = X_train.shape[0]

        #generate random retraining points
        k = self.number_retrainings
        n = stream.n_remaining_samples()
        d = retrain_size

        sample = random.sample(range(n-(k-1)*(d-1)), k)
        indices = sorted(range(len(sample)), key=lambda i: sample[i])
        return_indices = sorted(indices, key=lambda i: indices[i])
        values = [s + (d-1)*r for s, r in zip(sample, return_indices)]
        values.sort()
        #values = [4370, 4620, 4813, 5099, 5421, 5951, 6179, 6580, 7001,]

        print(self.name, ': ', values)

        results = execute_stream_no_drift_detection(stream, model, values, refit_use_Xtrain, X_train, y_train,
                                                    retrain_size, prediction_type=prediction_type)

        predictions = results['predictions']
        true_values = results['true_values']
        uncertainties = results['uncertainties']
        class_probabilities = results['class_probabilities']
        X_retrain = results['X_retrain']
        retraining_counter = results['retraining_counter']


        if prediction_type == 'regression':
            predictions = np.hstack(predictions)
        if prediction_type == 'classification':
            predictions = np.vstack(predictions)
            class_probabilities = predictions
            predictions = np.argmax(predictions, axis=1)

        true_values = np.hstack(true_values)
        #class_probabilities = np.vstack(class_probabilities)
        uncertainties = pd.Series(np.hstack(uncertainties))

        detected_drifts = []


        metrics, errors = compute_metrics(prediction_type, true_values, predictions, class_probabilities, targets,
                                          uncertainties, detected_drifts, retraining_counter)

        raw_results = {'uncertainties': uncertainties, 'tv': true_values, 'tv_cat': None, 'preds': predictions, 'probs': class_probabilities, 'errors': errors}

        return metrics, detected_drifts, raw_results

class Equal_Distribution():
    def __init__(self, number_retrainings):
        self.name = 'equal_distribution'
        self.number_retrainings = number_retrainings

    def run_stream(self, stream, model, X_train, y_train, retrain, prediction_type, retrain_after, refit_use_Xtrain,
                   ks_features, targets):
        retrain_size = int((stream.n_remaining_samples() / 0.850) * 0.01)
        train_size = X_train.shape[0]

        # generate random retraining points
        k = self.number_retrainings
        n = stream.n_remaining_samples()
        d = retrain_size

        # divide n by number of retrainings + 2
        space = int(np.round(n / (k+1), 0))
        values = [i*space for i in range(1,k+1)]
        values.sort()
        # detected_drifts = copy.deepcopy(values)
        print(values)

        # Execute stream logic
        results = execute_stream_no_drift_detection(stream, model, values, refit_use_Xtrain, X_train, y_train,
                                                    retrain_size, prediction_type=prediction_type)

        predictions = results['predictions']
        true_values = results['true_values']
        uncertainties = results['uncertainties']
        class_probabilities = results['class_probabilities']
        X_retrain = results['X_retrain']
        retraining_counter = results['retraining_counter']

        if prediction_type == 'regression':
            predictions = np.hstack(predictions)
        if prediction_type == 'classification':
            predictions = np.vstack(predictions)
            class_probabilities = predictions
            predictions = np.argmax(predictions, axis=1)

        true_values = np.hstack(true_values)
        # class_probabilities = np.vstack(class_probabilities)
        uncertainties = pd.Series(np.hstack(uncertainties))

        detected_drifts = []

        metrics, errors = compute_metrics(prediction_type, true_values, predictions, class_probabilities, targets,
                                          uncertainties, detected_drifts, retraining_counter)

        raw_results = {'uncertainties': uncertainties, 'tv': true_values, 'tv_cat': None, 'preds': predictions,
                       'probs': class_probabilities, 'errors': errors}

        return metrics, detected_drifts, raw_results

class No_Retraining():
    def __init__(self):
        self.name = 'no_retraining'

    def run_stream(self, stream, model,  X_train, y_train, retrain, prediction_type, retrain_after, refit_use_Xtrain, ks_features, targets):

        predictions = []
        true_values = []
        uncertainties = []
        class_probabilities = []

        batch_size = stream.n_remaining_samples()

        X_test, y_test = stream.next_sample(batch_size)
        y_pred, y_pred_uncertainty = model.predict(X_test)

        if prediction_type == 'regression':
            predictions.append(y_pred[0])
        if prediction_type == 'classification':
            predictions.append(y_pred)
        true_values.append(y_test)
        uncertainties.append(y_pred_uncertainty)
        #class_probabilities.append(y_probabilities)

        if prediction_type == 'regression':
            predictions = np.hstack(predictions)
        if prediction_type == 'classification':
            predictions = np.vstack(predictions)
            class_probabilities = predictions
            predictions = np.argmax(predictions, axis=1)

        true_values = np.hstack(true_values)
        #class_probabilities = np.vstack(class_probabilities)
        uncertainties = pd.Series(np.hstack(uncertainties))

        detected_drifts = []
        retraining_counter = 0

        metrics, errors = compute_metrics(prediction_type, true_values, predictions, class_probabilities, targets,
                                          uncertainties, detected_drifts, retraining_counter)

        raw_results = {'uncertainties': uncertainties, 'tv': true_values, 'tv_cat': None, 'preds': predictions, 'probs': class_probabilities, 'errors': errors}

        return metrics, detected_drifts, raw_results


class Adwin_Error():
    def __init__(self, prediction_type = 'regression'):
        self.name = 'adwin_error'
        self.prediction_type = prediction_type
        self.delta = 0.002


    def gridsearch_adwin_parameter(self, stream, model, drifts_to_detect, targets, features=1, starting_value=-3):

        stream.restart()

        five_percent = int(stream.n_remaining_samples() * 0.05)
        X_train, y_train = stream.next_sample(five_percent)

        # fit algorithm
        model.fit(X_train, y_train)

        condition = False
        m = starting_value #-1
        n = starting_value #-1

        X_test, y_test = stream.next_sample(2 * five_percent)
        y_pred, y_pred_uncertainty = model.predict(X_test)

        error = self.prediction_error(y_test, y_pred)

        while condition == False:
            print('-------------------------')
            param_box_upper = 10 ** m
            param_box_lower = 10 ** (n - 1)

            # evaluate for param_box_upper
            drifts_upper = []
            adwin = ADWIN(param_box_upper)

            for i in range(len(error)):
                adwin.add_element(error[i])
                if adwin.detected_change():
                    #print('Change has been detected in data: - of index: ' + str(i))
                    drifts_upper.append(i)
                    adwin.reset()

            # evaluate for param_box_lower
            drifts_lower = []
            adwin = ADWIN(param_box_lower)

            for i in range(len(y_pred_uncertainty)):
                adwin.add_element(y_pred_uncertainty[i])
                if adwin.detected_change():
                    # print('Change has been detected in data: - of index: ' + str(i))
                    drifts_lower.append(i)
                    adwin.reset()

            print('Adwin Parameter: {} - Detected Drifts: {}'.format(param_box_upper, len(drifts_upper)))
            print('Adwin Parameter: {} - Detected Drifts: {}'.format(param_box_lower, len(drifts_lower)))

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

            elif (len(drifts_upper) < drifts_to_detect) & (len(drifts_lower) < drifts_to_detect) & (
                    param_box_upper == 0.1):
                print('case3')

                parameter = 0.002
                condition = True


            elif (len(drifts_upper) > drifts_to_detect) & (len(drifts_lower) < drifts_to_detect):
                print('case4')

                m -= 0.05
                n += 0.05

        print('Final ADWIN Error paramter: ', parameter)
        self.delta = parameter




    def run_stream(self, stream, model,  X_train, y_train, retrain, prediction_type, retrain_after, refit_use_Xtrain,
                   ks_features, targets):

        retrain_size = int((stream.n_remaining_samples()/0.85)*0.01)
        train_size = X_train.shape[0]

        adwin = ADWIN(delta=self.delta)
        print(adwin.get_info())

        predictions = []
        true_values = []
        uncertainties = []
        class_probabilities = []
        accepted_drifts = []
        detected_drifts = []
        X_retrain = []

        prediction_error = 0
        overall_i = 0
        next_retrain = -1
        retraining_counter = 0


        while stream.n_remaining_samples() > 1:

            # Standard batch size for computation
            batch_size = 1000

            # Check if remaining stream has still enough instances
            if (stream.n_remaining_samples() < batch_size):
                batch_size = stream.n_remaining_samples()

            X_test, y_test = stream.next_sample(batch_size)
            y_pred, y_pred_uncertainty = model.predict(X_test)

            if prediction_type == 'regression':
                predictions.append(y_pred[0])
            if prediction_type == 'classification':
                predictions.append(y_pred)
            true_values.append(y_test)
            uncertainties.append(y_pred_uncertainty)
            X_retrain.append(X_test)

            prediction_error = self.prediction_error(y_test, y_pred)



            for j in range(len(prediction_error)):

                adwin.add_element(prediction_error[j])

                if adwin.detected_change():

                    if not accepted_drifts:
                        if retrain==True:
                            next_retrain = overall_i + j + retrain_after
                    elif (overall_i + j)-accepted_drifts[-1] > retrain_after:
                        if retrain==True:
                            next_retrain = overall_i + j + retrain_after

                    if not accepted_drifts:
                        accepted_drifts.append(overall_i + j)
                    elif (overall_i + j)-accepted_drifts[-1]> retrain_after:
                        accepted_drifts.append(overall_i + j)

                    detected_drifts.append(overall_i + j)
                    adwin.reset()

                if (overall_i + j) == next_retrain:

                    retraining_counter += 1
                    if refit_use_Xtrain:

                        # Remove excess training data
                        _ = X_retrain.pop()
                        _ = true_values.pop()
                        _ = predictions.pop()
                        _ = uncertainties.pop()
                        #_ = class_probabilities.pop()

                        # Add data up to drift instead
                        X_retrain.append(X_test[:j, :])
                        true_values.append(y_test[:j])
                        if prediction_type == 'regression':
                            predictions.append(y_pred[0][:j])
                        if prediction_type == 'classification':
                            predictions.append(y_pred[:j,:])
                        uncertainties.append(y_pred_uncertainty[:j])
                        #class_probabilities.append(y_probabilities[:j])

                        # Stack training data
                        X_retrain_array = np.vstack(X_retrain)
                        true_values_array = np.hstack(true_values)

                        # Attention, need to adapt X_retrain
                        model.refit(np.concatenate([X_train, X_retrain_array[-retrain_size:, :]]),
                                    np.concatenate([y_train, true_values_array[-retrain_size:]]))
                        batch_size = j
                        next_retrain = -1

                        # Restart stream and set to correct position
                        stream.restart()
                        _, _ = stream.next_sample((3 * X_train.shape[0]) + (overall_i + j))
                        break
                    else:
                        model.refit(np.concatenate(X_retrain[-200:]), np.concatenate(true_values[-200:]))

            # Need to adapt overall_i
            overall_i += batch_size

        if prediction_type == 'regression':
            predictions = np.hstack(predictions)
        if prediction_type == 'classification':
            predictions = np.vstack(predictions)
            class_probabilities = predictions
            predictions = np.argmax(predictions, axis=1)

        true_values = np.hstack(true_values)
        #class_probabilities = np.vstack(class_probabilities)
        uncertainties = pd.Series(np.hstack(uncertainties))

        metrics, errors = compute_metrics(prediction_type, true_values, predictions, class_probabilities, targets,
                                          uncertainties, detected_drifts, retraining_counter)

        print('ADWIN Error: ', detected_drifts)

        raw_results = {'uncertainties': uncertainties, 'tv': true_values, 'tv_cat': None, 'preds': predictions,
                       'probs': class_probabilities, 'errors': errors}


        return metrics, detected_drifts, raw_results



    def prediction_error(self, y_test, y_pred):

        if self.prediction_type == 'regression':
            # print((y_test - y_pred)**2)
            prediction_error = np.sqrt((y_test - y_pred[0]) ** 2)

        if self.prediction_type == 'classification':
            predictions = np.argmax(y_pred, axis=1)
            prediction_error = np.absolute((y_test - predictions) * 1)
            # print(prediction_error)

        return prediction_error







class KS_Data():
    def __init__(self, number_retrainings, limit_drifts = True, detection_only = False, retrain_after = 0, window_size = 200, stat_size = 100):
        self.name = 'kswin'
        self.limit_drifts = limit_drifts
        self.detection_only = detection_only
        self.alpha = 0.005
        self.retrain_after = retrain_after
        self.window_size = window_size
        self.stat_size = stat_size
        self.number_retrainings = number_retrainings

    def set_parameter(self, alpha, retrain_after = 0):
        self.alpha = alpha
        self.retrain_after = retrain_after

    def gridsearch_kswin_parameter(self, stream, model, drifts_to_detect, targets, features=1, starting_value = -3):

        stream.restart()

        five_percent = int(stream.n_remaining_samples() * 0.05)

        X_train, y_train = stream.next_sample(five_percent)

        condition = False
        m = starting_value
        n = starting_value

        ks_features = features

        X_test, y_test = stream.next_sample(2 * five_percent)
        retrain_after = 50

        while condition == False:
            print('-------------------------')
            param_box_upper = 10 ** m #* 10
            param_box_lower = 10 ** (n - 1) #* 10

            print('KSWIN Parameter: {}'.format(param_box_upper))

            # evaluate for param_box_upper
            drifts_upper = []

            kswin_dict = {}
            for i in range(ks_features):
                key = str(i)
                kswin_dict[key] = KSWIN(alpha=param_box_upper, window_size=self.window_size, stat_size=self.stat_size,
                                        data=X_train[:, i::ks_features][0])

            p_values_upper = []
            detected_drifts_upper = []
            accepted_drifts_upper = []

            for i in range(len(y_test)):
                # Process stream via KSWIN and print detections

                for feat in range(ks_features):
                    batch = X_test[i][feat]
                    kswin_dict[str(feat)].add_element(batch)
                    detected, p_value = kswin_dict[str(feat)].detected_change()
                    if detected:

                        detected_drifts_upper.append(i)

                        if not accepted_drifts_upper:
                            accepted_drifts_upper.append(i)
                            p_values_upper.append(p_value)

                        elif i - accepted_drifts_upper[-1] > self.retrain_after:
                            #print('Drift detection: {}'.format(i))
                            accepted_drifts_upper.append(i)
                            p_values_upper.append(p_value)

                        # kswin_dict[str(feat)].reset()
                        # print('Reset KSWIN')
                        # print(kswin_dict[str(feat)].get_info())

            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            print(current_time)
            print('Start lower KSWIN')

            # Lower
            kswin_dict = {}
            for i in range(ks_features):
                key = str(i)
                kswin_dict[key] = KSWIN(alpha=param_box_lower, window_size=self.window_size, stat_size=self.stat_size,
                                        data=X_train[:, i::ks_features][0])

            p_values_lower = []
            detected_drifts_lower = []
            accepted_drifts_lower = []

            for i in range(len(y_test)):
                # Process stream via KSWIN and print detections

                for feat in range(ks_features):
                    batch = X_test[i][feat]
                    kswin_dict[str(feat)].add_element(batch)
                    detected, p_value = kswin_dict[str(feat)].detected_change()
                    if detected:
                        detected_drifts_lower.append(i)

                        if not accepted_drifts_lower:
                            accepted_drifts_lower.append(i)
                            p_values_lower.append(p_value)

                        elif i - accepted_drifts_lower[-1] > self.retrain_after:
                            accepted_drifts_lower.append(i)
                            p_values_lower.append(p_value)

                        # kswin_dict[str(feat)].reset()
                        # print('Reset KSWIN')
                        # print(kswin_dict[str(feat)].get_info())

            print('KSWIN Parameter: {} - Detected Drifts: {}'.format(param_box_upper, len(accepted_drifts_upper)))
            print('KSWIN Parameter: {} - Detected Drifts: {}'.format(param_box_lower, len(accepted_drifts_lower)))
            print('Time: ', time.time())

            if len(accepted_drifts_upper) == drifts_to_detect:
                print('case1')
                parameter = param_box_upper
                condition = True

            elif len(accepted_drifts_lower) == drifts_to_detect:
                print('case1')
                parameter = param_box_lower
                condition = True


            elif (len(accepted_drifts_upper) > drifts_to_detect) & (len(accepted_drifts_lower) > drifts_to_detect):
                print('case2')
                m -= 1
                n -= 1

            elif (len(accepted_drifts_lower) < drifts_to_detect) & (len(accepted_drifts_lower) < drifts_to_detect) & (
                    param_box_upper == 0.1):
                print('case3')

                parameter = 0.00001
                condition = True


            elif (len(accepted_drifts_upper) > drifts_to_detect) & (len(accepted_drifts_lower) < drifts_to_detect):
                print('case4')

                m -= 0.05
                n += 0.05

        print('Final KSWIN paramter: ', parameter)
        self.alpha = parameter

        return parameter



    def run_stream(self, stream, model,  X_train, y_train, retrain, prediction_type, retrain_after, refit_use_Xtrain,
                   ks_features, targets):

        retrain_size = int((stream.n_remaining_samples()/0.850)*0.01)
        train_size = X_train.shape[0]



        print('KSWIN params: alpha: {}, window_size: {}, stat_size: {}'.format(self.alpha, self.window_size, self.stat_size))

        # Initialize KSWIN and a data stream
        kswin_dict = {}
        for i in range(ks_features):
            key = str(i)
            kswin_dict[key] = KSWIN(alpha=self.alpha, window_size = self.window_size, stat_size=self.stat_size,
                                    data=X_train[:,i::ks_features][0])

        p_values = []
        detected_drifts = []
        accepted_drifts = []

        X_test, y_test = stream.next_sample(stream.n_remaining_samples())

        for i in range(len(y_test)):
        # Process stream via KSWIN and print detections

            for feat in range(ks_features):
                batch = X_test[i][feat]
                kswin_dict[str(feat)].add_element(batch)
                detected, p_value = kswin_dict[str(feat)].detected_change()
                if detected:
                    detected_drifts.append(i)
                    print('Drift: {}'.format(i))

                    if not accepted_drifts:
                        accepted_drifts.append(i)
                        p_values.append(p_value)

                    elif i-accepted_drifts[-1]> self.retrain_after:
                        accepted_drifts.append(i)
                        p_values.append(p_value)

        if self.limit_drifts == True:
            indices = np.argsort(np.array(p_values))[:self.number_retrainings]
            retraining_points = np.sort(np.array(accepted_drifts)[indices])
            retraining_points = retraining_points.tolist()
            detected_drifts_kswin = retraining_points.copy()
            detected_drifts = retraining_points

        else:
            detected_drifts_kswin = accepted_drifts

        print('KSWIN: ', detected_drifts_kswin)

        #start evaluation
        stream.restart()
        retrain_size = int(stream.n_remaining_samples()*0.01)
        _, _ = stream.next_sample(15*retrain_size)

        if (self.detection_only == False):
            # Execute stream logic
            results = execute_stream_no_drift_detection(stream, model, detected_drifts_kswin, refit_use_Xtrain, X_train, y_train,
                                                        retrain_size, prediction_type=prediction_type)

            predictions = results['predictions']
            true_values = results['true_values']
            uncertainties = results['uncertainties']
            class_probabilities = results['class_probabilities']
            X_retrain = results['X_retrain']
            retraining_counter = results['retraining_counter']

            if prediction_type == 'regression':
                predictions = np.hstack(predictions)
            if prediction_type == 'classification':
                predictions = np.vstack(predictions)
                class_probabilities = predictions
                predictions = np.argmax(predictions, axis=1)

            true_values = np.hstack(true_values)
            #class_probabilities = np.vstack(class_probabilities)
            uncertainties = pd.Series(np.hstack(uncertainties))

            metrics, errors = compute_metrics(prediction_type, true_values, predictions, class_probabilities, targets,
                                              uncertainties, detected_drifts, retraining_counter)

            raw_results = {'uncertainties': uncertainties, 'tv': true_values, 'tv_cat': None, 'preds': predictions,
                           'probs': class_probabilities, 'errors': errors}

        else:
            metrics = {}
            raw_results = {}

        return metrics, detected_drifts, raw_results