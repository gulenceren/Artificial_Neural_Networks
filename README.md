Artificial Neural Network (ANN);
   We read the dataset with pandas and transfer it to an array. We converted it to numpy array format and shuffled the data with shuffle. Data is added to the x and y values from the zero elements of X to 1 less than the last colums of df. The last colums were added as y. 25 was split as test 75 train and train_test_split operation was applied. MLP will repeat this process 3 times and the hidden layer is divided as 100.

Deep Neural Network (DNN);
   1. With the menu we created, we do the dnn operations via h2o and keras, read our datasets, assign the colums named classifion as y with h2o and set the epoch numbers as 100,200,300 in the hypers params method. The necessary parameters were entered for 3 hidden layers to be hidden layers and 2 trials. The best models were ranked using deeplearning with H2O's GridSerach. Then the acc of the best model was found.

     2. Train and test data were read and the partitioning process was performed with iloc according to the number of features determined over these data. Cycles were provided with the Categorical structure. As long as the Acc value was less than 0.62, it was looped and then processes were carried out using 6 optimizers. A single hidden layer with 16 neurons was used and an output with 2 neurons was determined. By doing 150 epochs, the best result and model were recorded in 10 trials.
     
Convolutıonal Neural Networks (CNN);
    1. The ./Images folder, which should be in the same location as our application file, has been set as the path. As categories, the names of the 3 image folders in this path were determined as categories.
    2. These images were rearranged to be 70 size again and settings were made with grayscale. Train and test sequences were created according to the class number structure.
    3. This data was mixed and X and y were created according to the category structures on train, and test_X and test_y were created on test. Resized with reshape. Re-recorded with Pickle and opened the files back.
    4. Type transformation was done. Since there are 3 categories, categorial operation was done. While creating the model, the model was created as input as X.shape[1:], 2 hidden layers with 64 neurons and 3 outputs.
    5. The model that gives the best result was saved using 3 optimizers.

h20, xlwt, opencv yüklenmeli
