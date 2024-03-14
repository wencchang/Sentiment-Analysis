#%% 載入套件
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import nltk
nltk.download('stopwords')
from sklearn.model_selection import KFold

#%% 讀取 csv 檔後取前 1 萬筆資料
df=pd.read_csv('C:/Users/User/Desktop/IMT/Data_mining/HW2-Sentiment_Analysis/archive/Reviews.csv')
df_t = df.head(10000)

#%% 僅保留"Text"、"Score"兩個欄位
df_t = df_t.loc[:,['Score','Text']]

#%% 將 "Score" 欄位內值大於等於4的轉成1，其餘轉成0
#0: negative
#1: positive
df_t['Score'] = df_t['Score'].apply(lambda x: 1 if x >= 4 else 0)
print(df_t['Score'].value_counts())

#%% 切分文本
df_t['Text'] = df_t['Text'].str.split()
print(df_t.head())

#%% 去除停頓詞
stop_words = set(stopwords.words('english'))
df_t['Text'] = df_t['Text'].apply(lambda x: [item for item in x if item not in stop_words])
print("Text Data after Removing Stopwords:\n", df_t['Text'].head())

#%% TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(smooth_idf=True)
X_tfidf = tfidf_vectorizer.fit_transform(df_t['Text'].apply(' '.join))
y = df_t['Score']
print("TF-IDF Matrix:\n", X_tfidf.toarray())

#%% 建模 - 使用RandomForestClassifier
model_tfidf = RandomForestClassifier(n_estimators=200, max_depth=10, max_leaf_nodes=10)

#%% 進行k-fold cross-validation
scores_tfidf = cross_val_score(model_tfidf, X_tfidf, y, cv=4)
mean_accuracy_tfidf = scores_tfidf.mean()

print("Mean Accuracy (TF-IDF):", mean_accuracy_tfidf)

#%% Word2Vec
sentences = df_t['Text'].tolist()
w2v_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
X_w2v = []
for sentence in sentences:
    vec = np.zeros(100)
    count = 0
    for word in sentence:
        if word in w2v_model.wv:
            vec += w2v_model.wv[word]
            count += 1
    if count != 0:
        vec /= count
    X_w2v.append(vec)
X_w2v = np.array(X_w2v)
print("Word2Vec Matrix:\n", X_w2v)


#%% 建模 - 使用RandomForestClassifier
model_w2v = RandomForestClassifier(n_estimators=200, max_depth=10, max_leaf_nodes=10)


#%% 進行k-fold cross-validation
kf = KFold(n_splits=4, shuffle=True, random_state=42)
accuracies = []
for train_index, test_index in kf.split(X_w2v):
    X_train, X_test = X_w2v[train_index], X_w2v[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model_w2v.fit(X_train, y_train)
    y_pred = model_w2v.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
mean_accuracy_w2v = np.mean(accuracies)
print("Mean Accuracy (Word2Vec):", mean_accuracy_w2v)

