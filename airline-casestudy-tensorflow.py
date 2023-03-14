# %% [markdown]
# # Machine Learning- Deep Neural Network DNN

# %% [markdown]
# ## Explore the dataset
# 
# In continuing after the first script of the PACE stage, I've exported the cleaned csv file for running TensorFlow and develop a deep learning ML model, once loading the dataset, it should contain 129,880 customer data, with 21 important features for the airline service offering, and equipped with the satisfaction level, either "satisfied" or "dissatisfied".
# 
# > **Citation**: The airline dataset used in the this exercise is derived from the data science community platform which is Kaggle, the dataset has been provided by one of the Kaggle community providers named [MOHAIMEN AL RASHID](https://www.kaggle.com/datasets/mohaimenalrashid/invistico-airline). Besides, you may also obtain this dataset from Github platform as well, which is provided by [taukirazam16](https://github.com/taukirazam16).
# 

# %%
!pip install pandas
# %%
#!pip install --upgrade tensorflow

# %%
!pip install numpy

# %%
import numpy as np

# %%
import pandas as pd

# load the training dataset (excluding rows with null values)
airline = pd.read_csv('./airline-ml.csv').dropna()

# Deep Learning models work best when features are on similar scales
# In a real solution, we'd implement some custom normalization for each feature, but to keep things simple
# In the first script, I've scaled the dataset, therefore, there is no need to rescale again, 
# we just need to recheck the dataset by using describe() or head() function

airline.head() 

# %% [markdown]
# The **satisfaction** column is the label our model will predict. Each label value represents customer satisfaction, encoded as either 0 or 1. 

# %%
samples = airline[['seat_comfort', 'dept_arr_time_convenient','food_service', 
            'gate_location', 'inflight_wifi','inflight_entertainment', 
            'online_support', 'ease_online_booking','onboard', 
            'legroom', 'baggage_handling', 'checkin', 
            'cleanliness','online_boarding','satisfaction',]]
samples.head()

# %%
from sklearn.model_selection import train_test_split

features = ['seat_comfort', 'dept_arr_time_convenient','food_service', 
            'gate_location', 'inflight_wifi','inflight_entertainment', 
            'online_support', 'ease_online_booking','onboard', 
            'legroom', 'baggage_handling', 'checkin', 
            'cleanliness','online_boarding']
label = 'satisfaction'

# %%
satisfaction = ['Dissatisfied', 'Satisfied']
print(samples.columns[0:14].values, 'Services Rating')
for index, row in samples.sample(10).iterrows():
    print('[',row[0], row[1], row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13], row[14], ']',satisfaction[int(row[-1])])

# %% [markdown]
# As is common in a supervised learning problem, we'll split the dataset into a set of records with which to train the model, and a smaller set with which to validate the trained model.

# %%
   
# Split data 70%-30% into training set and test set
x_train, x_test, y_train, y_test = train_test_split(airline[features].values,
                                                    airline[label].values,
                                                    test_size=0.30,
                                                    random_state=0)

print ('Training Set: %d, Test Set: %d \n' % (len(x_train), len(x_test)))
print("Sample of features and labels:")

# Take a look at the first 25 training features and corresponding labels
for n in range(0,24):
    print(x_train[n], y_train[n], '(' + satisfaction[y_train[n]] + ')')

# %% [markdown]
# The *features* are the measurements for each customer rating observation, and the *label* is a numeric value that indicates the customer satisfaction that the observation represents (0= dissatisfied; 1= satisfied).
# 
# ## Install and import TensorFlow libraries
# 
# Since we plan to use TensorFlow to create our penguin classifier, we'll need to run the following two cells to install and import the libraries we intend to use.
# 
# > **Note** *Keras* is an abstraction layer over the base TensorFlow API. In most common machine learning scenarios, you can use Keras to simplify your code.


# %%
import tensorflow
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
from tensorflow.keras import optimizers

# Set random seed for reproducability
tensorflow.random.set_seed(0)

print("Libraries imported.")
print('Keras version:',keras.__version__)
print('TensorFlow version:',tensorflow.__version__)

# %% [markdown]
# ## Prepare the data for TensorFlow
# 
# We've already loaded our data and split it into training and validation datasets. However, we need to do some further data preparation so that our data will work correctly with TensorFlow. Specifically, we need to set the data type of our features to 32-bit floating point numbers, and specify that the labels represent categorical classes rather than numeric values.

# %%
# Set data types for float features
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Set data types for categorical labels
y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)
print('Ready...')

# %% [markdown]
# ## Define a neural network
# 
# Now we're ready to define our neural network. In this case, we'll create a network that consists of 3 fully-connected layers:
# * An input layer that receives an input value for each feature (in this case, the 14 features) and applies a *ReLU* activation function.
# * A hidden layer that receives ten inputs and applies a *ReLU* activation function.
# * An output layer that uses a *SoftMax* activation function to generate an output for customer satisfaction, with probability values in vector that sum to 1.

# %%
# Define a classifier network
hl = 10 # Number of hidden layer nodes

model = Sequential()
model.add(Dense(hl, input_dim=len(features), activation='relu'))
model.add(Dense(hl, input_dim=hl, activation='relu'))
model.add(Dense(len(satisfaction), input_dim=hl, activation='softmax'))

print(model.summary())

# %% [markdown]
# ## Train the model
# 
# To train the model, we need to repeatedly feed the training values forward through the network, use a loss function to calculate the loss, use an optimizer to backpropagate the weight and bias value adjustments, and validate the model using the test data we withheld.
# 
# To do this, we'll apply an Adam optimizer to a categorical cross-entropy loss function iteratively over 50 epochs.

# %%
#hyper-parameters for optimizer
learning_rate = 0.001
opt = optimizers.Adam(lr=learning_rate)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Train the model over 50 epochs using 10-observation batches and using the test holdout dataset for validation
num_epochs = 50
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=10, validation_data=(x_test, y_test))

 
# ## Review training and validation loss
# 
# After training is complete, we can examine the loss metrics we recorded while training and validating the model. We're really looking for two things:
# * The loss should reduce with each epoch, showing that the model is learning the right weights and biases to predict the correct labels.
# * The training loss and validation loss should follow a similar trend, showing that the model is not overfitting to the training data.
# 
# Let's plot the loss metrics and see:

# %%
%matplotlib inline
from matplotlib import pyplot as plt

epoch_nums = range(1,num_epochs+1)
training_loss = history.history["loss"]
validation_loss = history.history["val_loss"]
plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()

# %% [markdown]
# As seen from the above loss function graph in comparison of both training and cross-validation data, we may observed that the loss function of CV measures has peaked close to the epcoh of 25th and 40th. Ideally, despite of the projection of a smooth curve line of the CV, in the real world dataset like this Invistico Airline, since the loss validation was optimized directly from the training set of data, it is considered a normal projection of the losses measures has been detected. In overall, the validated data has projected a decline trend as aligned to the training sets, whereby the losses has been minimized across the epoches. 

# %% [markdown]
# ## View the learned weights and biases
# 
# The trained model consists of the final weights and biases that were determined by the optimizer during training. Based on our network model we should expect the following values for each layer:
# * Layer 1: There are 14 input values going to ten output nodes, so there should be 4 x 10 weights and 10 bias values.
# * Layer 2: There are ten input values going to ten output nodes, so there should be 10 x 10 weights and 10 bias values.
# * Layer 3: There are ten input values going to three output nodes, so there should be 10 x 2 weights and 2 bias values.

# %%
for layer in model.layers:
    weights = layer.get_weights()[0]
    biases = layer.get_weights()[1]
    print('------------\nWeights:\n',weights,'\nBiases:\n', biases)

# %% [markdown]
# ## Evaluate model performance
# 
# Once we're getting the accuracy report for the validation data,  the results appear to have much stronger weights and biases value. To dive in deeper into the view of each prediction performance of satisfaction class,we may apply *confusion matrix* model that shows a crosstab of correct and incorrect predictions for the satisfaction class.

# %%
# Tensorflow doesn't have a built-in confusion matrix metric, so we'll use SciKit-Learn
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
%matplotlib inline


class_probabilities = model.predict(x_test)
predictions = np.argmax(class_probabilities, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Plot the confusion matrix
cm = confusion_matrix(true_labels, predictions)
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(satisfaction))
plt.xticks(tick_marks, satisfaction, rotation=85)
plt.yticks(tick_marks, satisfaction)
plt.xlabel("Predicted Satisfaction")
plt.ylabel("Actual Satisfaction")
plt.show()

# %% [markdown]
# ## Save the trained model
# Once the results of the confusion matrix showed a strong diagonal line with darker shade for the correct prediction than the incorrect ones,  it is considered as likely an accurate model this case. Hence, we can save the model together with the trained weights for later use.

# %%
# Save the trained model
modelFileName = 'models/airline-classifier.h1'
model.save(modelFileName)
del model  # deletes the existing model variable
print('model saved as', modelFileName)

# %% [markdown]
# ## Use the trained model
# 
# Lastly, let's input a new customer data observation to test the model, whether the model to predict can correctly predict the satisfaction.
# In this case, my input feature is [5,4,5,2,2,5,3,2,5,4,4,5,5,5], the results was 'satisfied', seems accurate to me!

# %%
# Load the saved model
model = models.load_model(modelFileName)

# CReate a new array of features
x_new = np.array([[5,4,5,2,2,5,3,2,5,4,4,5,5,5]])
print ('New sample: {}'.format(x_new))

# Use the model to predict the class
class_probabilities = model.predict(x_new)
predictions = np.argmax(class_probabilities, axis=1)

print(satisfaction[predictions[0]])


