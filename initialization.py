# Essential libraries for model and tokenization
import keras
import pickle


# Essential libraries for text preprocessing
import re
from nltk.tokenize import TweetTokenizer, RegexpTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

# File paths for keras tokenizer and sentiment model
TOKENIZER_FILE = "model_files/tweet_tokenizer.pickle"
MODEL_FILE = "model_files/bidirectional_LSTM_layer_model_finalized.h5"

# Loading tokenizer and model
with open(TOKENIZER_FILE, 'rb') as handle:
    keras_tokenizer = pickle.load(handle)
sentiment_model = keras.models.load_model(MODEL_FILE)

# Preloading the preprocessing tokenizers
tweet_tokenizer = TweetTokenizer(strip_handles=True)
punct_tokenizer = RegexpTokenizer(r'\w+')
detokenizer = TreebankWordDetokenizer()

def preprocess_data(data): 
    
    global tweet_tokenizer, punct_tokenizer, detokenizer
    
    #Removing URLs with a regular expression
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    data = url_pattern.sub(r'', data)

    # Remove Emails
    data = re.sub('\S*@\S*\s?', '', data)

    # Remove new line characters
    data = re.sub('\s+', ' ', data)

    # Remove distracting single quotes
    data = re.sub("\'", "", data)
    
    # Remove capitalization of words
    data = data.lower()
    
    # Removal of twitter handles
    word_list =  tweet_tokenizer.tokenize(data)
    data = detokenizer.detokenize(word_list)
    
    # Remove punctuation
    filtered_punct_words = punct_tokenizer.tokenize(data)
    data = detokenizer.detokenize(filtered_punct_words)
    
    
    return data





