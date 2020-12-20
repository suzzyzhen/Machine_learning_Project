# Machine_learning_Project

Dataset
   ./OnlineNewsPopularity/OnlineNewsPopularity.csv  ##Base Data
    ./names.csv                                      ##Feature Information
    ./functions.py                                   ##Resource Functions for cleanliness


PURPOSE: 

-To perform Normalization and Train-Test split on input data, maintaining class distribution
-To train Neural Network methodically with Train data and predict on Test data

INSTRUCTIONS:
Go to NN.py

1. Define your desired architecture with: 
Arch=[25,15,1]

2. Train the model using: 
NN(Inputs, Labels, Epochs, BatchSize, LearningRate, Architecture)
ab = N(xTrain, ytrain, 800, 2000, 0.01, Arch) => these are the parameters for optimal accuracy

3. To test accuracy on the model, call the Accuracy function: 
Accuracy (In, Out, Weights, Biases)
acc = Accuracy(xTest, yTest, ab[0], ab[5]) 


Running the code returns the accuracy and the runtime. 

