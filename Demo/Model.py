from sklearn.linear_model import LogisticRegression

# Import the IRIS Dataset to be used in this Kernel
from sklearn.datasets import load_iris

# Load the Module to split the Dataset into Train & Test
from sklearn.model_selection import train_test_split

import pickle

Iris_data = load_iris()
Xtrain, Xtest, Ytrain, Ytest = train_test_split(Iris_data.data,
                                                Iris_data.target,
                                                test_size=0.3,
                                                random_state=4)

#Define the model
LR_Model = LogisticRegression(C=0.1,
                               max_iter=20,
                               fit_intercept=True,
                               n_jobs=3,
                               solver='liblinear')

# Train the Model
LR_Model.fit(Xtrain, Ytrain)

Pkl_Filename = "SavedModel.pkl"

with open(Pkl_Filename, 'wb') as file:
    pickle.dump(LR_Model, file)

# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:
    Pickled_LR_Model = pickle.load(file)

Pickled_LR_Model

score = Pickled_LR_Model.score(Xtest, Ytest)
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))

# Predict the Labels using the reloaded Model
Ypredict = Pickled_LR_Model.predict(Xtest)

Ypredict