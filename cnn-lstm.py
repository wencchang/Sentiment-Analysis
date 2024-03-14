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
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.layers import LSTM
import matplotlib.pyplot as plt
# %matplotlib inline
#%% 讀取 csv 檔後取前 1 萬筆資料
df=pd.read_csv("C:/Users/User/OneDrive/桌面/IMT/DataMining/HW4 - sentiment analysis 2/Reviews.csv")
df_all = df.head(10000)
#%% 僅保留"Text"、"Score"兩個欄位
df_all = df_all.loc[:,['Score','Text']]
#%% 將 "Score" 欄位內值大於等於4的轉成1，其餘轉成0
#0: negative
#1: positive
df_all['Score'] = df_all['Score'].apply(lambda x: 1 if x >= 4 else 0)
print(df_all['Score'].value_counts())
#%% 切分文本
df_all['Text'] = df_all['Text'].str.split()
print(df_all.head())
#%% 去除停頓詞
stop_words = set(stopwords.words('english'))
df_all['Text'] = df_all['Text'].apply(lambda x: [item for item in x if item not in stop_words])
print("Text Data after Removing Stopwords:\n", df_all['Text'].head())


#%%
df_train, df_test = train_test_split(df_all, test_size=0.3)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
# %%
x_train = df_train['Text'].tolist()
y_train = np.asarray( df_train['Score'] ).astype(np.float32)
x_test = df_test['Text'].tolist()
y_test = np.asarray( df_test['Score'] ).astype(np.float32)
# %% 建立token字典
token = Tokenizer(num_words=5000)
token.fit_on_texts(x_train) #排序在前 num_words 名的詞列入字典
#token.document_count
#token.word_index
# %% 使用token將評論轉換為數字list *[input_text]
x_train = token.texts_to_sequences(x_train)
x_test = token.texts_to_sequences(x_test)
# %% 截長補短讓所有數字list長度都一樣
x_train = pad_sequences(x_train, maxlen=500)
x_test = pad_sequences(x_test, maxlen=500)


# %% CNN without Dropout
#建立線性堆疊模型 將各神經網路層加入模型
modelCNN = Sequential()

#Embedding層將數字list轉換為向量list
modelCNN.add(Embedding(output_dim=100,
                       input_dim=5000,
                       input_length=500))

#卷積層1
modelCNN.add(Conv1D(filters=50, #幾個濾鏡
                    kernel_size=2, #濾鏡大小 N-gram
                    activation='relu'))

#池化層1
modelCNN.add(MaxPooling1D(pool_size=2))

#平坦層
modelCNN.add(Flatten())

#隱藏層
modelCNN.add(Dense(units=50,
                   activation='relu'))

#輸出層
modelCNN.add(Dense(units=1,
                   activation='sigmoid'))

modelCNN.summary()
# # %%
def run_train_history(model):
    
    #定義訓練方式
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    #開始訓練
    train_history = model.fit(x_train, y_train, validation_split=0.2,
                              batch_size=100, #批次筆數(每次訓練多少批次: 總訓練資料/批次筆數)
                              epochs=10, #訓練週期數(10筆accuracy與loss)
                              verbose=2) #顯示訓練過程            
    return train_history
# %%
def show_train_history(train_history, train, validation, save_path=None):
    
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    
    plt.savefig(save_path, dpi=100, bbox_inches='tight')  # 儲存圖片
    plt.show()

# %%
modelCNN_th = run_train_history(modelCNN)
# %%
show_train_history(modelCNN_th, 'accuracy', 'val_accuracy', save_path='C:/Users/User/OneDrive/桌面/IMT/DataMining/HW4 - sentiment analysis 2/cnn_acc.png')
show_train_history(modelCNN_th, 'loss', 'val_loss', save_path='C:/Users/User/OneDrive/桌面/IMT/DataMining/HW4 - sentiment analysis 2/cnn_loss.png')

# # %% 評估模型準確率
scores = modelCNN.evaluate(x_test, y_test, verbose=1)
scores[1]


# %% CNN with Dropout
modelCNN_D = Sequential()

modelCNN_D.add(Embedding(output_dim=100,
                         input_dim=5000,
                         input_length=500))
modelCNN_D.add(Dropout(0.7))

modelCNN_D.add(Conv1D(filters=50,
                      kernel_size=2,
                      activation='relu'))

modelCNN_D.add(MaxPooling1D(pool_size=2))

modelCNN_D.add(Flatten())

modelCNN_D.add(Dense(units=50,
                     activation='relu'))
modelCNN_D.add(Dropout(0.7))

modelCNN_D.add(Dense(units=1,
                     activation='sigmoid'))

modelCNN_D.summary()
# %%
modelCNN_D_th = run_train_history(modelCNN_D)
# %%
show_train_history(modelCNN_D_th, 'accuracy', 'val_accuracy', save_path='C:/Users/User/OneDrive/桌面/IMT/DataMining/HW4 - sentiment analysis 2/cnn_dropout_acc.png')
show_train_history(modelCNN_D_th, 'loss', 'val_loss', save_path='C:/Users/User/OneDrive/桌面/IMT/DataMining/HW4 - sentiment analysis 2/cnn_dropout_loss.png')
scores = modelCNN_D.evaluate(x_test, y_test, verbose=1)
scores[1]



# %% LSTM without Dropout
modelLSTM = Sequential()

modelLSTM.add(Embedding(output_dim=100,
                        input_dim=5000,
                        input_length=500))

#LSTM層
modelLSTM.add(LSTM(30))

modelLSTM.add(Dense(units=50,
                    activation='relu'))

modelLSTM.add(Dense(units=1,
                    activation='sigmoid'))

modelLSTM.summary()
# %%
modelLSTM_th = run_train_history(modelLSTM)
# %%
show_train_history(modelLSTM_th, 'accuracy', 'val_accuracy', save_path='C:/Users/User/OneDrive/桌面/IMT/DataMining/HW4 - sentiment analysis 2/lstm_acc.png')
show_train_history(modelLSTM_th, 'loss', 'val_loss', save_path='C:/Users/User/OneDrive/桌面/IMT/DataMining/HW4 - sentiment analysis 2/lstm_loss.png')

scores = modelLSTM.evaluate(x_test, y_test, verbose=1)
scores[1]



# %% LSTM with Dropout
modelLSTM_D = Sequential()

modelLSTM_D.add(Embedding(output_dim=100,
                          input_dim=5000,
                          input_length=500))
modelLSTM_D.add(Dropout(0.7))

modelLSTM_D.add(LSTM(30))

modelLSTM_D.add(Dense(units=50,
                      activation='relu'))
modelLSTM_D.add(Dropout(0.7))

modelLSTM_D.add(Dense(units=1,
                      activation='sigmoid'))

modelLSTM_D.summary()
# %% 
modelLSTM_D_th = run_train_history(modelLSTM_D)
# %%
show_train_history(modelLSTM_D_th, 'accuracy', 'val_accuracy', save_path='C:/Users/User/OneDrive/桌面/IMT/DataMining/HW4 - sentiment analysis 2/lstm_dropout_acc.png')
show_train_history(modelLSTM_D_th, 'loss', 'val_loss', save_path='C:/Users/User/OneDrive/桌面/IMT/DataMining/HW4 - sentiment analysis 2/lstm_dropout_loss.png')

scores = modelLSTM_D.evaluate(x_test, y_test, verbose=1)
scores[1]


#%%
df_test=pd.read_csv("C:/Users/User/OneDrive/桌面/IMT/DataMining/HW4 - sentiment analysis 2/test.csv")
df_test['Text'] = df_test['Text'].str.split()
print(df_test.head())
stop_words = set(stopwords.words('english'))
df_test['Text'] = df_test['Text'].apply(lambda x: [item for item in x if item not in stop_words])
print("Text Data after Removing Stopwords:\n", df_test['Text'].head())
df_test = df_test.reset_index(drop=True)
test = df_test['Text'].tolist()
# %% 使用token將評論轉換為數字list *[input_text]
test = token.texts_to_sequences(test)
# %% 截長補短讓所有數字list長度都一樣
test = pad_sequences(test, maxlen=500)

#%%test_cnn
pred_cnn = modelCNN_D.predict(test)
print("cnn : ", pred_cnn)
result_cnn = []
for i in range(len(pred_cnn)):
    a = pred_cnn[i][0]
    
    if a > 0.5:
        a = 1
    else:
        a = 0
        print(a)
    result_cnn.append(a)
print(result_cnn)
df_ans = pd.DataFrame(columns=['score'],data=result_cnn)
df_ans['ID'] = range(1, len(df_ans) + 1)
df_ans = df_ans[['ID', 'score']]
df_ans.to_csv('C:/Users/User/OneDrive/桌面/IMT/DataMining/HW4 - sentiment analysis 2/result_cnn.csv',index=False)

#%%test_lstm
pred_lstm = modelLSTM_D(test) 
print("lstm : ", pred_lstm)
result_lstm = []
for i in range(len(pred_lstm)):
    a = pred_lstm[i][0]
    
    if a > 0.5:
        a = 1
    else:
        a = 0
        print(a)
    result_lstm.append(a)
print(result_lstm)
df_ans = pd.DataFrame(columns=['score'],data=pred_lstm)
df_ans['ID'] = range(1, len(df_ans) + 1)
df_ans = df_ans[['ID', 'score']]
df_ans.to_csv('C:/Users/User/OneDrive/桌面/IMT/DataMining/HW4 - sentiment analysis 2/result_lstm.csv',index=False)

# %%
