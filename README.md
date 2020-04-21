# Movie Rating Prediction (Using ANN)

- **Domain** : Artificial Intelligence, Machine Learning
- **Sub-Domain** : Supervised Learning, Classification


## Libraries required:

1. Scikit-learn
2. Matplotlib
3. numpy
4. pandas
5. nltk
6. keras
7. Tensorflow


## Dataset

- Kaggle Dataset: https://www.kaggle.com/c/movie-review-sentiment-analysis/data

- The folder movie_data contains two files `train.txt` and `test.txt`.
    - *train.txt* contains 25000 reviews - one review in each line. The first 12500 reviews are labelled as *pos* while the next 12500 reviews are labelled as *neg*.
    - *test.txt* contains 25000 reviews without labels.


## Data Pre-processing

- Lowering of characters
- Removal of tags
- Cleaning
- Stemming
- Normalization

## Model

![Model_Summary](./Model_Summary.png?raw=true "Model_Summary")


## Approach
- Developed artificial neural network model for predicting movie rating from 25000 movie reviews.
- Performed data analysis, visualization, feature extraction, cleaning, preprocessing, feature transformation (one hot encoding) and trained with cross-validation.
- The main task was to predict the rating (positive or negative) of a movie based on its review. This was considered as a binary classification problem.
- The result of test dataset is saved in the file *y_pred.csv*.


## Instructions to run:

- `python3 Movie_Rating_Prediction_ANN.py `


## Results

- Acuuracy achieved using Artificial Neural Network: 90.24 %