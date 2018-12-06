# CODE

- NARM.py -- Main code -- Run this to train models - Needs Theano
- data_process.py -- Is called by NARM.py to load data from the .pkl train and test dataset files and to convert that data to numpy arrays and lists
- example_prepocess.py -- Standalone piece of code used to obtain the .pkl train and test dataset files from the original dataset published at http://cikm2016.cs.iupui.edu/cikm-cup/


# DATA

All of the data preprocessing code is agnostic to Theano except https://github.com/prashant-jayan21/sessionRec_NARM/blob/master/data_process.py#L32. You may want to edit it as per your needs when you switch to using TensorFlow.

The final train and datasets to be used are:
- data/digi_train.pkl
- data/digi_test.pkl

(The validation set is created on-the-fly in the code.)

Also, the original raw dataset can be found at data_raw/dataset-train-diginetica.


# CODING GUIDELINES

Please do not push to master if your changes are unstable. Always work on a separate branch and merge to master whenever ready.




