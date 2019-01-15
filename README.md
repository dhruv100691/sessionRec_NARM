# CODE

- model.py -- Main code -- Run this to train models 
- data_process.py -- Is called by NARM.py to load data from the .pkl train and test dataset files and to convert that data to numpy arrays and lists
- example_prepocess.py -- Standalone piece of code used to obtain the .pkl train and test dataset files from the original dataset published at http://cikm2016.cs.iupui.edu/cikm-cup/


# DATA

The final train and datasets to be used are:
- data/digi_train.pkl
- data/digi_test.pkl

(The validation set is created on-the-fly in the code.)

Also, the original raw dataset can be found at data_raw/dataset-train-diginetica.

The items feature matrix can be found here: https://drive.google.com/drive/folders/1uLCH3xGEu6LA0AC6weyHsRZqEeM4DiFF?usp=sharing.

**Format:**
- #rows = #items (**NOTE: row index = item id - 1**)
- #columns = feature vector size = #unique prices + #unique categories

Each feature vector is a concatenation of the following two vectors:
- A one-hot vector for price info (size = 12)
- A one-hot vector for category info (size = 995)

You can also experiment with picking just one of these feature vectors instead of using the concatenated one. You'll need to split the numpy matrix accordingly.





