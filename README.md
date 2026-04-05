# Ch-Mamba
The full implementation code for Ch-Mamba code. It includes all the codes in order to use the model to train and test the model using  multivariate time series forecasting datasets. The code as it is, will work with the following dataset https://zenodo.org/records/15089283. 

The full training code is on FULLTRAINING.py

The full testing code is on FULLTESTING.py

You can also build these codes manually as explained below.

## Training Procedure
In order to build the code for training the model, put the following codes in the following order:
1. Imports.py
2. (Colab) Upload dataset
3. Preprocessing.py 
4. Model.py
5. Arguments.py
6. Training.py
7. (Optional) Epochs.py

## Testing Procedure
If you are planning to use the pretrained model on the dataset mentioned put the following codes in order:
1. Imports.py
2. (Colab) Upload files.py here upload the args.pth, scaler.pkl, weights.pth
