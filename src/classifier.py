from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re,nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words=set(stopwords.words('english'))
lemmatizer=WordNetLemmatizer()

def preprocess(text):
    #preprocessing the data (converting to lowercase,removing special char,etc)
    text=text.lower()
    text=re.sub(r'^\w\s]','',text)
    tokens=word_tokenize(text)
    tokens=[lemmatizer.lematize(i) for i in tokens if i not in stop_words]
    return " ".join(tokens)
class IntentClassifier:
    def __init__(self):
        self.vectorizer= CountVectorizer()
        self.model=MultinomialNB()

    def train(self,texts,labels):
        X=self.vectorizer.fit_transform(texts)
        self.model.fit(X,labels)
    def predict(self,text):
        X=self.vectorizer.transform([text])
        return self.model.predict(X)[0]
    
    def fit_and_predict(self,texts,labels,text_predict):
        self.train(texts,labels)
        return self.predict(text_predict)
    
