#Meet JIA, a simple chatbot for simple queries

#importing necessary libraries
import io
import random
import string 
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True)

#uncomment the following only the first time
#nltk.download('punkt') #first-time use only
#nltk.download('wordnet') #first-time use only

#Reading in the corpus
with open('chatbot.txt','r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()

#Tokenisation
sent_tokens = nltk.sent_tokenize(raw) #converts to list of sentences 
word_tokens = nltk.word_tokenize(raw) #converts to list of words

#Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#Keyword Matching
GREET_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREET_RESPONSES = ["Hi", "Hey", "*nods*", "Hey there", "Hello", "I am glad! You are talking to me"]

def greeting(sentence):
    """If user's input is a greet, return a greet response"""
    for word in sentence.split():
        if word.lower() in GREET_INPUTS:
            return random.choice(GREET_RESPONSES)

#Generating response
def response(user_response):
    jia_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        jia_response=jia_response+"I am sorry! I didn't get you"
        return jia_response
    else:
        jia_response = jia_response+sent_tokens[idx]
        return jia_response

flag = True
print("JIA: My name is Jia. I will answer your queries about Chatbots. If you want to exit, type 'bye'!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if (user_response!='bye'):
        if (user_response=='thanks' or user_response=='thank you' ):
            flag = False
            print("JIA: You are welcome")
        else:
            if (greeting(user_response)!=None):
                print("JIA: "+greeting(user_response))
            else:
                print("JIA: ", end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("JIA: Bye! Take care")
        
        

