import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import pickle
import random
import json

try:
    nltk.download('punkt')
except:
    pass

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:  # rb = read bytes, storing workspace so we don't  have to run except for every input.
        words, labels, training, output = pickle.load(f)  # storing these 4 variables into our pickle file.
except:
    words = []  # contains non duplicated words from patterns.
    labels = []  # contains tags or labels
    docs_x = []  # contains list of all the different patterns
    docs_y = []  # list for element keeping track of pattern which patterns it is a part of
    # loading all words and labels and getting all documents ready with different patterns.

    for intents in data["intents"]:  # looping through all of the dictionaries (intents)
        for pattern in intents["patterns"]:  # from each intent getting patterns dictionary
            wrds = nltk.word_tokenize(pattern)  # separate our words, stemming.
            words.extend(wrds)  # put these words in words list
            docs_x.append(wrds)  # add words in docs
            docs_y.append(intents["tag"])

        if intents["tag"] not in labels:  # if tag(greetings) not in labels list,
            labels.append(intents["tag"])  # a way of getting different greetings tag from intents dictionary

    # stemming words and removing duplicates. Stemming get the main word, like if we have whats, it will take "word" only
    # and ignore "s".
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]  # stemming for words list
    words = sorted(list(set(words)))  # removing duplicates by converting list to set, then we convert set to list back
    labels = sorted(labels)

    # for neural network we will create a list of ints where each index will contain the 1 if it exists in
    # our pattern otherwise 0. This list will have same size as our number of words in a pattern.

    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]  # list with 0 initially for all classes("tag") and
    # if it exists we'll replace it with 1
    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]  # stemming words all words in our pattern, doc will get all doc from docs|_pattern

        for w in words:  # looping through original word list.
            if w in wrds:  # if word exists in the current pattern we're looping through.
                bag.append(1)
            else:
                bag.append(0)

        # generating output
        output_row = out_empty[:]  # making copy of out_empty list
        output_row[labels.index(docs_y[x])] = 1  # looking in labels list, setting 1 in output row
        # by looking where the tag is in that list.
        training.append(bag)  # contains bags of words (lists of 0s and 1s)
        output.append(output_row)  # also contains 0s and 1s

    # converting  training and output into numpy arrays for TF learn. converting lists in arrays, as they are int lists.
    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:  # saving workspace
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()  # reset to get rid of previous settings
net = tflearn.input_data(shape=[None, len(training[0])])  # define input shape for model, length of input = data length
net = tflearn.fully_connected(net, 8)  # adding fully-connected layer to our neural network
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")  # sofmax will give us probability of each
net = tflearn.regression(net)  # output result.

model = tflearn.DNN(net)  # DNN is a network, it'll take the network|(net) we created and use it.

try:  # opening model if its already trained. It saves the existing trained mode, else open model.
    model.load("model.tflearn")

except:
    model.fit(training, output, n_epoch=1500, batch_size=8,
              show_metric=True)  # n_epoch is amount of times model is gonna
    # see the same data.
    model.save("model.tflearn")


# generating bag of words.
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]  # blank bag of word lists, filled with 0.

    s_words = nltk.word_tokenize(s)  # list of tokenize words
    s_words = [stemmer.stem(word.lower()) for word in s_words]  # stemming.

    for se in s_words:  # se = word in sentence
        for i, w in enumerate(words):  # w = current word, current word we are looking at in our words list.
            if w == se:  # if  current word = word in sentence
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print("Start talking with the bot! ")
    print("to quit, type quit! ")
    while True:
        inp = input("User: ")  # what user will type.
        if inp.lower() == "quit":
            break
        results = model.predict([bag_of_words(inp, words)])[0]  # model.predict is used for making predictions.
        # we are passing our bag of words function for predictions and saving it in a variable.

        # print(results)  , this will give us probabilities of all answers(neuron) from a class(tag).

        results_index = numpy.argmax(results)  # gives index of greatest value(probability) from our list.
        tag = labels[results_index]  # this will give us the label of our message
        if results[results_index] > 0.7:
            #print(tag)
            for tg in data["intents"]:      # tg = tag,
               if tg['tag'] == tag:        #   if label tag = intents tag, get responses from intents tag.
                  responses = tg['responses']    # responses in a dictionary in intents file.

            print(random.choice(responses))     # randomly selects one of the responses.
        else:
            print("I don't quite understand, try a different question.")

chat()
