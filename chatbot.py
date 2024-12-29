# import random
# import json
# import pickle
# import numpy as np
#
# import nltk
# from nltk.stem import WordNetLemmatizer
#
# from tensorflow.keras.models import load_model
#
# lemmatizer = WordNetLemmatizer()
# intents = json.load(open('intents.json'))
#
# words = pickle.load(open('words.pkl', 'rb'))
# classes = pickle.load(open('classes.pkl', 'rb'))
# model = load_model('chatbot_model.keras')
#
#
# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
# #Intelligent Web Chatbot
#     return sentence_words
#
#
# def bag_of_words(sentence):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for w in sentence_words:
#         for i, word in enumerate(words):
#             if word == w:
#                 bag[i] = 1
#     return np.array(bag)
#
#
# def predict_class(sentence):
#     bow = bag_of_words(sentence)
#     res = model.predict(np.array([bow]))[0]
#     ERROR_THRESHOLD = 0.25
#
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
#     results.sort(key=lambda x: x[1], reverse=True)
#
#     return_list = []
#
#     for r in results:
#         return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
#
#     return return_list
#
#
# def get_response(intents_list, intents_json):
#     tag = intents_list[0]['intent']
#     list_of_intents = intents_json['intents']
#
#     for i in list_of_intents:
#         if i['tag'] == tag:
#             result = random.choice(i['responses'])
#             break
#
#     return result
#
#
# while True:
#     message = input('')
#     ints = predict_class(message)
#     res = get_response(ints, intents)
#     print(res)

import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Load trained data and model
intents = json.load(open('intents.json'))
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.keras')

# Preprocess user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)  # Tokenize the sentence into words
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]  # Lemmatize words
    return sentence_words

# Convert user input into a bag-of-words representation
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)  # Initialize bag with 0s
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1  # Mark words present in the user input
    return np.array(bag)

# Predict the class of user input
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]  # Predict probabilities for each class
    ERROR_THRESHOLD = 0.25  # Filter predictions below this threshold

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)  # Sort by probability

    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})  # Map results to intents
    return return_list

# Fetch a response based on the predicted intent
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']  # Get the top intent
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])  # Randomly select a response
            break
    return result

# Main loop to interact with the chatbot
while True:
    message = input('You: ')
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(f"Bot: {res}")
