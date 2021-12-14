import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer_method_call = PorterStemmer()

def tokenize_user_message(message):
   
    return nltk.word_tokenize(message)


def stem_words(word):
   
    return stemmer_method_call.stem(word.lower())


def bag_of_words_data(tokenized_message, words):
   
    message_words = [stem_words(stemmed_word) for stemmed_word in tokenized_message]
    bag_data = np.zeros(len(words), dtype=np.float32)
    for idx, word_of_msg in enumerate(words):
        if word_of_msg in message_words: 
            bag_data[idx] = 1

    return bag_data