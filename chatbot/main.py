import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
#helps implement best practices for data automation, model tracking, performance monitoring, model retraining
import tensorflow
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)

#try to recall info from pickle
try:
    #throw in an x so it doesnt give old pickle data
    #where we save info, save as reasd bytes
    with open("data.pickle","rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    
    #for every intent (what question the text implies like name, greeting, age)
    for intent in data["intents"]:
        #for every pattern in an intent
        for pattern in intent["patterns"]:
            #gets every word in patterns 
            wrds = nltk.word_tokenize(pattern)
            #adds all the words in
            words.extend(wrds)
            #adds every pure/tokenized words
            docs_x.append(wrds)
            #every tag correlates to a pattern
            docs_y.append(intent["tag"])
            
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
    
    #makes the words readable in words (remove -ings, excess cutoffs), remove duplicate elements after
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    
    #optinal?
    labels = sorted(labels)
    
    #start creating the training and testing output (words and patterns)
    #creates a bag of words that represents all of the words in a a given pattern
    #known as one hot encoded, list is a length of amt of words
    #each encoding is a 1 0 entries with 0's and 1's that says if a word exists
    #https://www.google.com/search?q=neural+networks&sxsrf=ALiCzsYXyQXtsq79tLgaSQdorHXUCZVKCw:1667361691741&source=lnms&tbm=isch&sa=X&ved=2ahUKEwi2meSqzo77AhXtg2oFHej0BrMQ_AUoAXoECAIQAw&biw=1280&bih=673&dpr=1.5
    training = []
    output = []
    
    out_empty = [0 for _ in range(len(labels))]
    
    for x, doc in enumerate(docs_x):
        bag = []
        
        wrds = [stemmer.stem(w) for w in doc]
        
        #check if stemmed words matches stemmed tags
        #automatically add correlation if it does
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        
        output_row = out_empty[:]
        #look through tags and see where the tag is in the list and set it to 1 in output row (tells system what pattern the matching words correlate to)
        output_row[labels.index(docs_y[x])] = 1
        
        training.append(bag)
        output.append(output_row)
        
    #convert to np arrays for tflearn
    training = numpy.array(training)
    output = numpy.array(output)
    
    with open("data.pickle","wb") as f:
        pickle.dump((words, labels, training, output),f)
    
#default
tensorflow.compat.v1.reset_default_graph()

#define input shape we're expecting from model, each training "model" is 
net = tflearn.input_data(shape=[None, len(training[0])])
#add fully connected layer to neural network which has input data above and has 8 neurons for hidden layer
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
#gives probality for each neuron in the layer
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

#type of neural network to use after we defined 
model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    #passing training data and save the model
    #n_epoch is how many times it sees the data, batch size is how many times a model changes, showmetric prints out model
    model.fit(training, output, n_epoch=1000, batch_size = 8, show_metric = True)
    model.save("model.tflearn")

def bag_of_words(s,words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    
    for se in s_words:
        for i, w in enumerate(words):
            if(w == se):
                 bag[i] = 1
    return numpy.array(bag)

def chat():
    print("Start talking with the bot! (type quit to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        #how probable the model thinks its under a specific tag, and pick up the first
        results = model.predict([bag_of_words(inp, words)])[0]
        #gets the max result
        results_index = numpy.argmax(results)
        #stores labels
        tag = labels[results_index]
        #check the place, get out of it later
        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print(random.choice(responses))
        else:
            print("I don't understand")
    
chat()


















