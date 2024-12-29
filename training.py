# import os
# import random
# import json
# import pickle
#
# import numpy as np
# import nltk
# # nltk.download('punkt_tab')
# from nltk.stem import WordNetLemmatizer
#
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation, Dropout
# from tensorflow.keras.optimizers import SGD
#
# lemmatizer = WordNetLemmatizer()
#
# intents = json.load(open('intents.json'))
#
# words = []
# classes = []
# documents = []
# ignore_letters = ['?', '!', '.', ',']
#
# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         word_list = nltk.word_tokenize(pattern)
#         words.extend(word_list)
#         documents.append((word_list, intent['tag']))
#         if intent['tag'] not in classes:
#             classes.append(intent['tag'])
#
# words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
# words = sorted(set(words))
#
# classes = sorted(set(classes))
#
# pickle.dump(words, open('words.pkl', 'wb'))
# pickle.dump(classes, open('classes.pkl', 'wb'))
#
# training = []
# output_empty = [0] * len(classes)
#
# for document in documents:
#     bag = []
#     word_patterns = document[0]
#     word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
#
#     for word in words:
#         bag.append(1) if word in word_patterns else bag.append(0)
#
#     output_row = list(output_empty)
#     output_row[classes.index(document[1])] = 1
#     training.append([bag, output_row])
#
# random.shuffle(training)
# # Training data extraction
# training = np.array(training, dtype=object)  # Ensure dtype=object to handle lists of unequal length
#
# train_x = np.array([item[0] for item in training])  # List of feature vectors (bags of words)
# train_y = np.array([item[1] for item in training])  # List of one-hot encoded labels
#
#
# model = Sequential()
# model.add(Dense(512, input_shape=(len(train_x[0]),), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(train_y[0]), activation='softmax'))
#
# # Correct optimizer definition using 'learning_rate' instead of 'lr'
# sgd = SGD(learning_rate=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
#
# # Compiling the model
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# model.fit(np.array(train_x), np.array(train_y), epochs=500, batch_size=5, verbose=1)
#
# model.save('chatbot_model.keras')
import os


import random
import json
import pickle

import numpy as np
import nltk
# nltk.download('punkt_tab')  # Uncomment this line if 'punkt' is not already downloaded.
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

# Load intents file containing patterns and responses.
intents = json.load(open('intents.json'))

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Process each pattern in intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)  # Tokenize patterns into individual words
        words.extend(word_list)
        documents.append((word_list, intent['tag']))  # Append tokenized words and corresponding intent tag
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and remove duplicates for the word list
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))  # Sort intent classes alphabetically

# Save preprocessed data
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

# Create training data: bag-of-words vectors and one-hot encoded labels
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)  # Bag-of-words representation

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1  # Mark the corresponding intent as 1
    training.append([bag, output_row])

# Shuffle and convert training data into NumPy arrays
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array([item[0] for item in training])  # Features
train_y = np.array([item[1] for item in training])  # Labels

# Define neural network architecture
model = Sequential()
model.add(Dense(512, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))  # Output layer with softmax for multi-class classification

# Configure optimizer and compile the model
sgd = SGD(learning_rate=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
model.fit(np.array(train_x), np.array(train_y), epochs=500, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbot_model.keras')
