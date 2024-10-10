import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.data.path.append('C:/Users/user/AppData/Roaming/nltk_data')
nlp = spacy.load('en_core_web_sm')

def check_empty_message(msg):
    if not msg.strip():
        raise ValueError("The message can't be empty.")
    return msg


def process_message(msg):
    msg = msg.lower()
    msg = re.sub(r'[^\w\s]', '', msg)
    tokens = word_tokenize(msg)
    return tokens


def process_message_pipeline(msg):
    check_empty_message(msg)
    tokens = process_message(msg)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatized_tokens = [token.lemma_ for token in nlp(' '.join(tokens))]
    return lemmatized_tokens


#test_msgs = [
 #   "Hello, World! This is an example.",
  #  "I love programming in Python.",
   # "Chatbots can be very useful, don't you think?",
    #"The weather is great today.",
    #"   ",
    #"Running, jogging, and swimming are my favorite exercises."
#]

#for test_msg in test_msgs:
 #   try:
  #      output = process_message_pipeline(test_msg)
   #     print(f"Original Message: {test_msg}\nProcessed Output: {output}\n")
    #except ValueError as e:
     #   print(f"Original Message: {test_msg}\nError: {e}\n")
