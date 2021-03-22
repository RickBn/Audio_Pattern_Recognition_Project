# Riccardo Bona - Audio_Pattern_Recognition_Project: Music genre classification using GTZAN dataset

The project consists in the analysis of the performance of four different classification methods applied to music genre discrimination using the GTZAN dataset. 
The dataset contains 100 audio tracks for each one of the 10 represented music genres. From the audio tracks, 9 short-term audio features have been extracted and 
integrated with mean and standard deviation values. In order to analyze the complexity of the classification task, the feature space has been clustered using 
K-means. Classification has been performed using the following algorithms: K-nearest neighbors (K-NN), decision tree (DT), multi-layer perceptron (MLP) and support 
vector machines (SVM). The performance of each classification algorithm has been evaluated using repeated K-fold cross-validation and by comparing accuracy, precision, 
recall and F1-score for both training and test sets. Results highlight a much higher score on the training part for all classifiers. 
SVM and MLP achieved the best performance on the test part with overall accuracy scores of 75% and 77% respectively, the score for K-NN is around 69\% and DT performed 
the worst with a score of 53%.

The repository contains two folders:

- data: contains the files "features.csv" with the extracted audio features. 
- scripts: contains all the Python scripts used in the project.
    - "functions.py" contains all the custom made functions used in the other scripts;
    - "features_extraction.py" is the script that extracts the features from the dataset containing the actual audio tracks. 
       Note that the folder "data/genres_original/" is not included in the repository but can be downloaded at
       https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification
    - "clustering.py" is the script used to perform cluster analysis of the feature space;
    - "classification.py" is the script containing the code relative to the tested classification algorithms.
