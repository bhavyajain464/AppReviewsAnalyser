import json
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from google_play_scraper import Sort, reviews, app
import re
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import enchant
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gensim
from gensim.models import Word2Vec
import scipy
from gensim import matutils
import string
from nltk import word_tokenize, pos_tag
from sklearn.feature_extraction import text
d = enchant.Dict("en_US")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')



sns.set(style='whitegrid', palette='muted', font_scale=1.2)
app_infos = []
info = app("com.facebook.katana", lang='en', country='us')
del info['comments']
app_infos.append(info)


def print_json(json_object):
  json_str = json.dumps(
    json_object,
    indent=2,
    sort_keys=True,
    default=str
  )

app_infos_df = pd.DataFrame(app_infos)
app_infos_df.to_csv('apps.csv', index=None, header=True)

app_reviews = []

for score in tqdm(list(range(1, 2)),leave=True,position=0):
    for sort_order in [Sort.NEWEST]:
        rvs, _ = reviews(
        "com.facebook.katana",
        lang='en',
        country='us',
        sort=sort_order,
        count= 20000,
        filter_score_with=score
        )
        for r in rvs:
            r['sortOrder'] = 'most_relevant' if sort_order == Sort.MOST_RELEVANT else 'newest'
            r['appId'] = "com.facebook.katana"
        app_reviews.extend(rvs)

for score in tqdm(list(range(2, 3)),leave=True,position=0):
    for sort_order in [Sort.NEWEST]:
        rvs, _ = reviews(
        "com.facebook.katana",
        lang='en',
        country='us',
        sort=sort_order,
        count= 40000,
        filter_score_with=score
        )
        for r in rvs:
            r['sortOrder'] = 'most_relevant' if sort_order == Sort.MOST_RELEVANT else 'newest'
            r['appId'] = "com.facebook.katana"
        app_reviews.extend(rvs)

for score in tqdm(list(range(3, 6)),leave=True,position=0):
    for sort_order in [Sort.NEWEST]:
        rvs, _ = reviews(
        "com.facebook.katana",
        lang='en',
        country='us',
        sort=sort_order,
        count= 10000,
        filter_score_with=score
        )
        for r in rvs:
            r['sortOrder'] = 'most_relevant' if sort_order == Sort.MOST_RELEVANT else 'newest'
            r['appId'] = "com.facebook.katana"
        app_reviews.extend(rvs)


# print_json(app_reviews[0])
# print(len(app_reviews))
app_reviews_df = pd.DataFrame(app_reviews)
app_reviews_df.to_csv('reviews.csv', index=None, header=True)



data = pd.read_csv("reviews.csv")
apps = pd.read_csv("apps.csv")
data = data[['userName','at','content','score','appId']]
data['length'] = data['content'].astype(str).apply(len)
data['label'] = (data['score']/3).apply(int)
data['length'].plot(kind = 'hist',x='num_children',y='num_pets')
plt.savefig('lentgh.png')
plt.close()


#Light Preprocessing


lemmatizer = WordNetLemmatizer()
def text_process(k):
    k=k.encode('ascii', 'ignore').decode('ascii')
    clean_string = re.sub(r"\W"," ",k)
    clean_string = clean_string.replace("_", " ")
    clean_string = re.sub(r"\d"," ",clean_string)
    clean_string = re.sub(r"\s+[a-z]\s+"," ",clean_string,flags=re.I)
    clean_string = re.sub(r"\s+"," ",clean_string)
    clean_string = re.sub(r"^\s"," ",clean_string)
    clean_string = re.sub(r"\s$"," ",clean_string)
    clean_string.lower()
    tokens = nltk.word_tokenize(clean_string)
    tokens = [token.lower() for token in tokens if token.lower() not in stopwords.words('english') and d.check(token)]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

data['clean_content'] = np.array(list(map(text_process, data['content'].values.astype('U'))))
data['length'] = data['clean_content'].astype(str).apply(len)
data = data[data['length']>0]
data['label'] = (data['score']/3).apply(int)
data['score'].plot(kind = 'hist',bins=10)
plt.savefig('light preprocessing.png')
plt.close()


#Heavy Preprocessing

analyser = SentimentIntensityAnalyzer()

review_drop = []
for i in range(len(data)):
  score = analyser.polarity_scores(data['clean_content'].iloc[i])
  if score['compound']>-0.05:
    if data['label'].iloc[i]==0:
      review_drop.append(i)
  else:
    if data['label'].iloc[i]==1:
      review_drop.append(i)
rows = data.index[review_drop]
data = data.drop(rows)

# import keras
# from keras.preprocessing.text import Tokenizer

text_tokens = [text.split(" ") for text in data['clean_content'].values.tolist()]
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(data['clean_content'].values.tolist())
# word2id = tokenizer.word_index
# id2word = dict([(value,key) for (key,value) in word2id.items()])

# # print(word2id)
# vocab_size  =len(word2id)+1
# print(vocab_size)
# embedding_dim = 100
# max_len = 150
# X = [[word2id[word] for word in sent] for sent in text_tokens]

# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
# pad = 'post'
# X_pad = pad_sequences(X,maxlen = max_len,padding = pad,truncating=pad)
# print(X_pad[2])
# label2id = {l:i for i,l in enumerate(set(data['label']))}
# id2label = {v:k for k,v in label2id.items()}
# y = [label2id[label] for label in data['label']]
# y = to_categorical(y,num_classes=len(label2id),dtype = 'float32')
# print(X_pad.shape,y.shape)

# from keras.models import Model
# from keras.layers import Input,Embedding,Dropout,Bidirectional,LSTM,TimeDistributed,Dense,Activation,Dot,Reshape
# seq_input = Input(shape = (max_len,),dtype = 'int32')
# embedded = Embedding(vocab_size,embedding_dim,input_length=max_len)(seq_input)
# embedded = Dropout(0.2)(embedded)
# lstm = Bidirectional(LSTM(embedding_dim,return_sequences = True))(embedded)
# lstm = Dropout(0.2)(lstm)
# att_vector = TimeDistributed(Dense(1))(lstm)
# att_vector = Reshape((max_len,))(att_vector)
# att_vector = Activation('softmax',name = 'attention_vec')(att_vector)
# att_output = Dot(axes = 1)([lstm,att_vector])
# fc = Dense(embedding_dim,activation='relu')(att_output)
# output = Dense(len(label2id),activation='softmax')(fc)
# model = Model(inputs = [seq_input],outputs = output)
# model.summary()

# model.compile(loss="categorical_crossentropy",metrics = ['accuracy'],optimizer = 'adam')
# history = model.fit(X_pad,y,epochs=2,batch_size=64,validation_split=0.2,shuffle=True,verbose = 1)

# print(text_tokens)


data['score'].plot(kind = 'hist',bins=10)
plt.savefig('heavy preprocessing.png')
plt.close()

#LDA for topics
model1 = gensim.models.Word2Vec(text_tokens, min_count = 1,vector_size = 100, window = 5) 
cv = CountVectorizer()
data_cv = cv.fit_transform(data['clean_content'])
tdm = data_cv.transpose()
sparse_counts = scipy.sparse.csr_matrix(tdm)
corpus = matutils.Sparse2Corpus(sparse_counts)
id2word = dict((v, k) for k, v in cv.vocabulary_.items())

def nouns(text):
    is_noun = lambda pos: pos[:2] == 'NN'
    tokenized = word_tokenize(text)
    all_nouns = [word for (word, pos) in pos_tag(tokenized) if     is_noun(pos)]
    return ' '.join(all_nouns)

data_nouns = pd.DataFrame(data['clean_content'].apply(nouns))


add_stop_words = ['like', 'im', 'know', 'just', 'dont', 'thats', 'right', 'people','youre', 'got', 'gonna', 'time', 'think', 'yeah', 'said']
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)
cvn = CountVectorizer(stop_words=stop_words)
data_cvn = cvn.fit_transform(data_nouns['clean_content'])
data_dtmn = pd.DataFrame(data_cvn.toarray(), columns=cvn.get_feature_names())
data_dtmn.index = data_nouns.index

corpusn=matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmn.transpose()))
id2wordn = dict((v, k) for k, v in cvn.vocabulary_.items())


def fun(s):
    return s[7:-1]
    
topics =[]

for i in range(3,6):
  ldan = gensim.models.LdaModel(corpus=corpusn, num_topics=i, id2word=id2wordn, passes=10)
  for topic in ldan.show_topics():
    topics.append(' '.join(list(map(fun, topic[1].split(" + "))))+" \n")

file1 = open("topics.txt","w+")
file1.writelines(topics)
file1.close()
# print(topics)

# def process(k,p):
#   ans = model1.wv.wmdistance(k,topics[p])
#   return float(ans)


# for i in range(len(topics)):
#   data['topic'+str(i)] = data['clean_content'].apply(process,p=i)
#   data.sort_values('topic'+str(i),ascending=False)[data['topic'+str(i)]!=np.inf][:1000]['score'].plot(kind = 'hist',bins=10)
#   # plt.text(0, 42000, topic[i], fontsize=15)
#   plt.savefig('topic'+str(i)+'.png')
#   plt.close()




