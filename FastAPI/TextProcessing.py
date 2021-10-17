import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import re
import pandas as pd


# creating a Preprocessing class containing all preprocessing required to the text data
class Preprocessing:
    '''Preprocesses the given column and returns data that is ready to feed to the model'''
    
    def __init__(self):
        # initializing objects for different preprocessing techniques
        self.stop_words = stopwords.words('english')
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2))
        self.selector = None
        self.training = None
        
    
    def remove_html_tags(self, text):
        # Remove html tags from a string
        html_free = re.compile('<.*?>')
        html_free = re.sub(html_free, '', text)
        return html_free


    def remove_punctuations(self, text):
        # removes unnecessary punctuations from the text
        text = text.lower()
        cleaned_text = re.findall("[a-zA-Z]+", text)
        
        return cleaned_text
    

    def stop_words_remover(self, text):
        # removes stopwords since they are not useful for sentiment prediction
        cleaned_text = [w for w in text if not w in self.stop_words]
        
        return cleaned_text
    
    
    def lemmatize(self, text):
        # brings words to their root words 
        cleaned_text = ' '.join([self.lemmatizer.lemmatize(i) for i in text])
        
        return cleaned_text
    
    
    def vectorize(self, X_cleaned): 
        # converts text data to vectorized form such that it can be feeded to the models
        if self.training:
            self.vectorizer.fit(X_cleaned)
            
        # converting text to vectorized form
        X_vectorized = self.vectorizer.transform(X_cleaned)
        
        return X_vectorized


    def feature_selection(self, X_vectorized, train_labels=None):
        # selects top 20K features by feature importance using f_classif
        if self.training:
            self.selector = SelectKBest(f_classif, k=min(20000, X_vectorized.shape[1]))
            self.selector.fit(X_vectorized, train_labels)
        
        # Select top 'k' of the vectorized features
        X_selected = self.selector.transform(X_vectorized).astype('float32')

        return X_selected
        
    
    def preprocess(self, X, train_labels=None, training=True):
        # takes input column and applies different pre-processing techniques
        X_cleaned = pd.DataFrame()
        self.training = training
        X = X.apply(lambda x: self.remove_html_tags(x))
        X = X.apply(lambda x: self.remove_punctuations(x))
        # X = X.apply(lambda x: self.stop_words_remover(x))
        X_cleaned = X.apply(lambda x: self.lemmatize(x))
        X_vectorized = self.feature_selection(self.vectorize(X_cleaned), train_labels)
        
        return X_cleaned, X_vectorized