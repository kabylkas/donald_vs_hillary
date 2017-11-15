import csv
import nltk
nltk.download("stopwords")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy
import math
import re
import pickle

def clean(sentence, stops):
  count=0
  sen = ""
  q = False
  e = False
  for word in sentence.split():
    word = word.replace(" ", "")

    if word.isdigit():
      count+=1
    if word.isdigit():
      word = ""
    sen+=word+" "

  filtered_sentence = [word for word in sen.split() if word not in stops]
  filtered_sentence = " ".join(filtered_sentence)
  filtered_sentence = re.sub('[!\;\:\_\-\*?.,\\\/()"\']', ' ', filtered_sentence)
  filtered_sentence = filtered_sentence.lower()
  filtered_sentence = filtered_sentence.decode('ascii', errors='ignore').encode()
  for i in range(count):
    filtered_sentence+=" **number**"
  return filtered_sentence
 
#read training set
t = []
corpus = []

with open('../input/train.csv', 'rb') as train_file:
  reader = csv.reader(train_file, delimiter=',', quotechar='|')
  for row in reader:
    #get the target value
    if (len(row)>1):
      if (row[0] == "handle"):
        continue
      if (row[0] == "HillaryClinton"):
        t.append(1)
      else:
        t.append(0)

      #get the text 
      corpus.append(row[1])

numpy.savetxt('./dumps/t.out', t)
#remove stopwords
filtered_corpus = []
stops = nltk.corpus.stopwords.words("english")
for sentence in corpus:
  filtered_sentence = clean(sentence, stops)
  filtered_corpus.append(filtered_sentence)

#build the dictionary
vectorizer = CountVectorizer(decode_error='ignore')
vectorizer.fit_transform(filtered_corpus)

#get the feature vectors of each message
transformer = TfidfTransformer(smooth_idf=False)
#init the training matrix
counts = vectorizer.transform(["you"]).toarray()
tfidf = transformer.fit_transform(counts)
train_matrix = numpy.matrix(tfidf.toarray())

for sentence in filtered_corpus:
  counts = vectorizer.transform([sentence]).toarray()
  tfidf = transformer.fit_transform(counts)
  train_matrix = numpy.vstack([train_matrix, tfidf.toarray()])
#delete the first row that was created on init
train_matrix = numpy.delete(train_matrix, (0), axis=0)
#transpose the training matrix
train_matrix = numpy.transpose(train_matrix)

numpy.savetxt('./dumps/train_matrix.out', train_matrix)
pickle.dump(vectorizer, open('./dumps/vectorizer.out', 'wb'))
pickle.dump(transformer, open('./dumps/transformer.out', 'wb'))
