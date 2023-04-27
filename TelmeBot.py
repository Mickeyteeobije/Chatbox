import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import tensorflow as tf
import random
import json
import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
#from sklearn.svm import SVC
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load the intents file
with open('assets/memory.json') as file:
    data = json.load(file)

# Extract the intents, patterns, and responses
intents = data['intents']
# Create lists to store the tokenized words, classes, and documents
words = []
classes = []
documents = []
ignore_chars = ['?', '.', ',']

# Tokenize the patterns and extract the words, classes, and documents
for intent in intents:
    for pattern in intent['patterns']:
        # Tokenize the pattern
        tokens = nltk.word_tokenize(pattern)
        # Remove punctuation
        tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token not in ignore_chars]
        # Add the words to the words list
        words.extend(tokens)
        # Add the document and the class to the documents and classes list
        documents.append((tokens, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Sort and remove duplicates from the words and classes lists
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
# Create the training data
training = []
output_empty = [0] * len(classes)
for document in documents:
    # Create a bag of words for each pattern
    bag = []
    pattern_words = document[0]
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
    # Create the output for each pattern
    output = list(output_empty)
    output[classes.index(document[1])] = 1
    training.append([bag, output])

# Shuffle the training data
random.shuffle(training)
training = np.array(training)

# Split the training data into input and output
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_y[0]), activation='softmax')])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model to the training data
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save('assets/telmed_model.h5')
# Load the saved model
model = tf.keras.models.load_model('assets/telmed_model.h5')

# Define a function to preprocess the input text
def preprocess_input(input_text):
    # Tokenize the input text
    tokens = nltk.word_tokenize(input_text)
    # Remove punctuation
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token not in ignore_chars]
    return tokens

# Define a function to predict the intent of the input text
def predict_intent(input_text):
    # Preprocess the input text
    tokens = preprocess_input(input_text)
    # Create a bag of words for the input text
    bag = [0] * len(words)
    for token in tokens:
        if token in words:
            bag[words.index(token)] = 1
    # Predict the class of the input text
    result = model.predict(np.array([bag]))[0]
    # Get the index of the predicted class
    index = np.argmax(result)
    # Get the class label of the predicted class
    intent = classes[index]
    # Choose a random response from the intents file
    for intent_data in intents:
        if intent_data['tag'] == intent:
            responses = intent_data['responses']
    response = random.choice(responses)
    return response

def greeting(sentence):
    """The Robot greets us using a random greeting"""

    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


#mapping strings to numbers
training = pd.read_csv('assets/Training.csv')
x = training.iloc[:, :-1]
y = training['prognosis']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=20)
clf = DecisionTreeClassifier()
X = x   
clf.fit(X_train, y_train)
cols= training.columns
cols= cols[:-1]
le = preprocessing.LabelEncoder()
le.fit(y)
reduced_data = training.groupby(training['prognosis']).max()

def readn(nstr):
    engine = pyttsx3.init()

    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)

    engine.say(nstr)
    engine.runAndWait()
    engine.stop()


severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index
def calc_condition(exp,days):
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp)+1)>13):
        print("You should take the consultation from doctor. ")
    else:
        print("It might not be that bad but you should take precautions.")


def getDescription():
    global description_list
    with open('assets/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open('assets/symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('assets/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)

def check_pattern(dis_list,inp):
    pred_list=[]
    inp=inp.replace(' ','_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list=[item for item in dis_list if regexp.search(item)]
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,[]
def sec_predict(symptoms_exp):
    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return clf.predict([input_vector])


def print_disease(node):
    node = node[0]
    val  = node.nonzero() 
    disease = le.inverse_transform(val[0])
    return list(map(lambda x:x.strip(),list(disease)))

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []

    while True:
        print("\nEnter the symptom you are experiencing or input 'help' to see possible symptom \t\t",end="->")
        disease_input = input("")
        conf,cnf_dis=check_pattern(chk_dis,disease_input)
        if disease_input.lower() == 'help':
            for index, element in enumerate(chk_dis):
                print(str(index)+')', element)
        else:
            if conf==1:
                print("searches related to input: ")
                for num,it in enumerate(cnf_dis):
                    print(num,")",it)
                if num!=0:
                    print(f"Select the one you meant (0 - {num}):  ", end="")
                    conf_inp = int(input(""))
                else:
                    conf_inp=0

                disease_input=cnf_dis[conf_inp]
                break
                print("Did you mean: ",cnf_dis,"?(yes/no) :",end="")
                conf_inp = input("")
                if(conf_inp=="yes"):
                     break
            else:
                print("Enter valid symptom.")

    while True:
        try:
            num_days=int(input("Okay. From how many days ? : "))
            break
        except:
            print("Enter valid input.")
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            # print( "You may have " +  present_disease )
            red_cols = reduced_data.columns 
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            dis_list=list(symptoms_present)
            print("symptoms given: ")
            for i in symptoms_given:
                print(i +' \t\t\t') 
            #print("symptoms given "  +  str(list(symptoms_given)) )
            print("Are you experiencing any ")
            symptoms_exp=[]
            for syms in list(symptoms_given):
                inp=""
                print(syms,"? : ",end='')
                while True:
                    inp=input("")
                    if(inp=="yes" or inp=="no"):
                        break
                    else:
                        print("provide proper answers i.e. (yes/no) : ",end="")
                if(inp=="yes"):
                    symptoms_exp.append(syms)

            second_prediction=sec_predict(symptoms_exp)
            # print(second_prediction)
            calc_condition(symptoms_exp,num_days)
            if(present_disease[0]==second_prediction[0]):
                print("You may have ", present_disease[0])
                print(description_list[present_disease[0]])

                # readn(f"You may have {present_disease[0]}")
                # readn(f"{description_list[present_disease[0]]}")

            else:
                print("You may have ", present_disease[0], "or ", second_prediction[0])
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])

            # print(description_list[present_disease[0]])
            precution_list=precautionDictionary[present_disease[0]]
            print("Take following measures : ")
            for  i,j in enumerate(precution_list):
                print(i+1,")",j)
    recurse(0, 1)

# Starting the chatbot
# Import the required module for text 
# to speech conversion
import pyttsx3
flag = True
print("-----------------------------------HealthCare ChatBot (Telmebot) -----------------------------------")
print("Telmebot: Welcome!!") 
# init function to get an engine instance for the speech synthesis
engine = pyttsx3.init()
engine.setProperty('voice', "english+f5")
engine.setProperty('rate', 135)
w = open('assets/welcome.txt','r',errors='ignore')
we= w.read()
# say method on the engine that passing input text to be spoken

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",'helo', 'h')
GREETING_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me"]

while(flag==True):
    user_response = input()
    user_response = user_response.lower()
    if(user_response !='bye'):
        if (user_response=='thanks' or user_response=='thank you'):
            flag=False
            print("Telmebot: You are welcome.")
        else:
            if (greeting(user_response)!= None):
                print("Telmebot: " + greeting(user_response))
                print("Telmebot: My name is Telmebot\n") 
                engine.say('{}, My name is Telme bot'.format(greeting(user_response)))
                #engine.say(we)
                print(we)
                engine.say('I will answer our queries about health related issues. If you want to exit, type  Bye!')
                print('\nI will answer our queries about health related issues. If you want to exit, type  Bye!')
                print("\nI can always check your health status at all time, just use the word: 'check me' ")
                engine.say(" I can always check your health status at all time, just use the word: 'check me' ")
                print('How may i help you!')
                # run and wait method, it processes the voice commands.
                engine.runAndWait()
            else:
                print("Telmebot: ", end="")
                print("Telmebot: ", predict_intent(user_response))
                #print("\nI can alway check you at all time, just use the word: 'check me' ")
                if user_response.lower() != 'check me':
                    print('Telmebot:  I\'m here at your service')
                    print("Remember, I can always check your health status at all time, just use the word: 'check me' ")
                    continue
                elif user_response == 'check me':
                    getSeverityDict()
                    getDescription()
                    getprecautionDict()  
                    tree_to_code(clf,cols)
                else:
                    print('Telmebot:  I\'m here at your service')
                    continue        

        
    else:
        flag=False
        print("Telme: Bye! Take care.")
