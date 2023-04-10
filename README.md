# Twitter Sentiment Analysis

This notebook examines a compendium of tweets collected from users on Twitter, with labels 0 and 1, denoting a negative and positive sentiment respectively. As part of an NLP project on Sentiment Analysis, several models are fitted to the dataset in an attempt to find the model that can best classify tweets as either Negative or Positive.

## Required File “IT1244” containing:
- **twitter_sentiment_analysis.ipynb** 
	- Jupyter Notebook containing the code
- **dataset.csv** 
	 - Dataset contained in a Comma Delimited text file
- **weights.h5** 
	- Weights for baseline LSTM Model
- **weights2.hdf5** 
	- Weights for baseline Bidirectional LSTM Model
- **model** 
	- Folder containing model saved from training Improved LSTM Model
- **glove.twitter.27B.100d.txt**
	 - GloVe Text file containing global vectors for word representation

**Ensure that the IT1244 File is saved into MyDrive of the Google Drive account connected to Colab after running the “Importing Libraries” chunk of code to ensure no errors!**
## Variables
There are only 2 variables:

- **Label**: Sentiment (1 = Positive, 0 = Negative)
- **Text**: Contents of tweet 

## Structure
The notebook is structured as follows:
1. Data preprocessing
2. Exploratory data analysis
3. Fitting of models to data
4. Evaluation of models

## Running the Notebook
### 1. Required Libraries:
- pandas
- numpy
- regex
- matplotlib
- seaborn
- plotly
- plotly_express
- nltk
- wordcloud
- sklearn
- xgboost
- tensorflow
- string
- contractions
- collections
- copy

Running the second cell will automatically install nltk packages required if they have not been installed already.
```
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
```

**The command to install new libraries is `pip install <library-name>`, but it may differ depending on your version of Python.**

### 2. Reading Dataset
Place the folder labeled “IT1244, into MyDrive of the google Drive account linked to Colab.

Then, run the following cell, which reads in the dataset as a pandas Data Frame:

```
df = pd.read_csv('drive/MyDrive/IT1244/dataset.csv', header = None)
df.columns = ['label', 'text']
```

If an error occurs here, it is likely that
a.) *pandas* was not imported correctly, or
b.) The dataset is not in the right directory. 

### 3. Data Preprocessing
From the Markdown chunk titled **Data Preprocessing** to the Markdown chunk titled **Exploratory Data Analysis**, ensure that all code chunks in between the two aforementioned Markdown chunks are ran. This ensures that the helper functions are loaded in, and that the dataset is cleaned. 
- The relevant comments can be found in the notebook.

### 4. Exploratory Data Analysis
Simply run all code chunks found.

### 5. Train Test Split and Vectorization for NLP
Ensure that the code chunk under **Train Test Split** is ran.
This is where the dataset is split into the respective training and testing set.
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state = 0) 
```

Next, ensure that the code chunk under **Vectorization for NLP** is ran.
This ensures that the dataset will be vectorized as per the code.
```
vectorizer = CountVectorizer() # converts text to token counts
X_train_v = vectorizer.fit_transform(X_train) # fits and vectorizes text in training set
X_test_v = vectorizer.transform(X_test) # vectorize text in testing set
```

**When in doubt, simply run all code chunks found in order.**

### 6. Modelling
Before starting, ensure that **weights.h5**, **weights2.hdf5** and the **model** folder are located in IT1244 Folder saved in Google Drive. These are the weights or model obtained from training the various LSTM models on the dataset, and allow you to skip running the process of fitting the model again which takes a significant amount of computational time.

For example,

**This chunk of code should be ran as it generates the model where the weights can be fed into.**
```
### THIS CHUNK OF CODE SHOULD BE RAN!
### This is used to build the neural network.
model = Sequential() # initialise sequential model
# add embedding layer to map input sequences to dense vectors
model.add(Embedding(input_dim = vocab_size, output_dim = output_dim, input_length = input_length))
# add dropout layer with regularization that randomly sets a fraction of input channels to zero thereby preventing overfitting
model.add(SpatialDropout1D(0.15))
# add LSTM layer
model.add(Bidirectional(LSTM(100, dropout = 0.15, recurrent_dropout = 0.15, return_sequences = True)))
model.add(Bidirectional(LSTM(100, dropout = 0.15, recurrent_dropout = 0.15, return_sequences = True)))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()```
```

**The following chunk of code can be skipped if desired as it simply generates the model weights have been found already.**
```
### THIS CHUNK OF CODE CAN BE SKIPPED IF DESIRED!
### This is used to fit the LSTM model to the training data
### Generates the weights "weights2.hdf5"
### Will take some time to run!
checkpoint_b = ModelCheckpoint('weights2.hdf5', monitor = 'loss', save_best_only = True) # save best weights from training
early_stop = EarlyStopping(monitor = 'loss', patience = 3) # stop training after 3 epochs if no improvement
history_b = model.fit(X_train, y_train, epochs = 25, batch_size = 64, callbacks = [checkpoint_b, early_stop],

use_multiprocessing = True)
```
**The following chunk of code should be ran such that the weights are loaded into the LSTM model.**
```
### THIS CHUNK OF CODE SHOULD BE RAN!
### This loads the weights you have placed in the directory.
model.load_weights('weights2.hdf5')
model.evaluate(X_test, y_test)
```


Next, ensure that the helper function, `get_scores`, is ran. It is located directly under the  **Modelling** Markdown chunk. This ensures that the scores for each model may be collected and stored.

```
### Helper function
def get_scores(y_test, y_pred):
    """
    Input: Actual Labels, Predicted Labels
    Output: Accuracy, Precision, F1-Score and Recall
    """
    acc = accuracy_score(y_test, y_pred) 
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return [acc, prec, f1, recall]
```

Here, you may run each model as desired. Each chunk has been separated by Markdown text that displays the model that will be built and fitted to the dataset in subsequent chunks.

For each model, you can expect to see the following:
- Initialising model
- Fitting model to training set
- Tuning of model
- Generation of evaluation metrics

Comments have been included where appropriate. Ideally, each model should be ran at least once, exclusive of tuning, such that the scores for each model may be collected. Otherwise, an error may occur in the subsequent section, *Model Evaluation*.

**Note: Tuning of each model takes a significant amount of computational time. You may choose to skip running the code chunks involving the tuning of the model which has already been commented out but can be ran by highlighting the commented out code chunks and using the shortcut Ctrl + /. Appropriate comments and annotations have been included, so they may be found easily.**

### 7. Model Evaluation
Simply run the  code chunk and a graph depicting the evaluation metrics for all models will be displayed.

**If you encounter any errors here such as being unable to generate the graph, it is likely that one or more of the models under the *Modelling* section have not been ran, therefore the evaluation metrics for that model have not been generated yet**.
