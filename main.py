from googleapiclient.discovery import build
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import re
from dotenv import load_dotenv
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import string
from string import punctuation
load_dotenv()

api_key = os.getenv('API_KEY')

class DataProcess:
    def __init__(self,api_key,video_id):
        self.api_key=api_key
        self.video_id=video_id
    
    def video_comments(self,comments=[]):
        # creating youtube resource object
        youtube = build('youtube', 'v3',
                developerKey=api_key)
        # retrieve youtube video results
        video_response=youtube.commentThreads().list(
        part='snippet,replies',
        videoId=video_id
        ).execute()
        # iterate video response
        while video_response:
                for item in video_response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']                               
                    replycount = item['snippet']['totalReplyCount']  
                    '''
                    if replycount > 0:
                        for reply in item['replies']['comments']:
                            reply_text = reply['snippet']['textDisplay']
                            replies.append(reply_text)
                    '''
                    comments.append(comment)
                    #comments += replies
                    
                    replies = []
                    if len(comments) > 10000:
                        return comments
               
                if 'nextPageToken' in video_response:
                    video_response = youtube.commentThreads().list(
                        part='snippet,replies',
                        videoId=video_id,
                        pageToken=video_response['nextPageToken']
                    ).execute()
                else:
                    return comments


video_id = "X0tOpBuYasI"
x=DataProcess(api_key,video_id).video_comments()
df = pd.DataFrame({'comments': x})
#nltk.download('vader_lexicon')

sentiments = SentimentIntensityAnalyzer()
df["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in df["comments"]]
df["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in df["comments"]]
df["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in df["comments"]]
df['Compound'] = [sentiments.polarity_scores(i)["compound"] for i in df["comments"]]
score = df["Compound"].values
sentiment = []

for i in score:
    if i >= 0.05 :
        sentiment.append('Positive')
    elif i <= -0.05 :
        sentiment.append('Negative')
    else:
        sentiment.append('Neutral')
df["Sentiment"] = sentiment

data2=df.drop(['Positive','Negative','Neutral','Compound'],axis=1)
#nltk.download('stopwords')
#nltk.download('wordnet')
stop_words = stopwords.words('english')
porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer() 
snowball_stemer = SnowballStemmer(language="english")
lzr = WordNetLemmatizer()

def text_processing(text):   
    text = text.lower()
    # remove new line characters in text
    text = re.sub(r'\n',' ', text)
    # remove punctuations from text
    text = re.sub('[%s]' % re.escape(punctuation), "", text)
    # remove references and hashtags from text
    text = re.sub("^a-zA-Z0-9$,.", "", text)
    # remove multiple spaces from text
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    # remove special characters from text
    text = re.sub(r'\W', ' ', text)

    text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])
    text=' '.join([lzr.lemmatize(word) for word in word_tokenize(text)])
    return text

#nltk.download('omw-1.4')
#nltk.download('punkt')
data_copy = data2.copy()
data_copy.comments = data_copy.comments.apply(lambda text: text_processing(text))

le = LabelEncoder()
data_copy['Sentiment'] = le.fit_transform(data_copy['Sentiment'])

processed_data = {
    'Sentence':data_copy.comments,
    'Sentiment':data_copy['Sentiment']
}

processed_data = pd.DataFrame(processed_data)

print(processed_data['Sentiment'].value_counts())

df_neutral = processed_data[(processed_data['Sentiment']==1)] 
df_negative = processed_data[(processed_data['Sentiment']==0)]
df_positive = processed_data[(processed_data['Sentiment']==2)]

# upsample minority classes
df_negative_upsampled = resample(df_negative, 
                                 replace=True,    
                                 n_samples= 4000, 
                                 random_state=42)  

df_neutral_upsampled = resample(df_neutral, 
                                 replace=True,    
                                 n_samples= 4000, 
                                 random_state=42)  

# Concatenate the upsampled dataframes with the neutral dataframe
final_data = pd.concat([df_negative_upsampled,df_neutral_upsampled,df_positive])
print(final_data['Sentiment'].value_counts())

corpus = []
for sentence in final_data['Sentence']:
    corpus.append(sentence)
corpus[0:5]

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = final_data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

nb_score = accuracy_score(y_test, y_pred)