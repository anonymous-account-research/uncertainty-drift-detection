Detecting Concept Drift With Neural Network Model Uncertainty

Authors: --- 

Please install requirements.txt before running code.

Experiments were conducted with Python 3.7.

This repository contains one folder per benchmark dataset.

- WrapperClasses_Classification.py: implementation of MCDropout as Classifier
- WrapperClasses_Regression.py: implementation of MCDropout as Regressor
- Detection_Strategies.py: implementation of detection strategies ADWINUncertainty, NoRetraining, Uninformed, KSWIN, ADWINError
- KSWIN.py: Adapted KSWIN-method from scikit-multiflow

Each folder contains:
(- Dataset)
(- Results_file)

- util.py: helper functions for benchmark
- Dataset_Performance_Evaluation.py: Benchmark test for a dataset with results
