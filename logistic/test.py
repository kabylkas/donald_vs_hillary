# stochastic gradient descent
# accuracy=0.913719393704 
#
# lumping all numbers into single string
# accuracy=0.930042751652
#
# adding "**questions**" and "**exclamation**" reduced accuracy
# accuracy=0.910610182666 
import csv
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy
import math
import matplotlib as plt
import pickle
import re
import random

#helper function
def clean(sentence, stops):
  count=0
  sen = ""
  q = False
  e = False
  for word in sentence.split():
    word = word.replace(" ", "")
    if "?" in word:
      q = True
    if "!" in word:
      e = True

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

def sigmoid(z):
  x = float(1)/(1+numpy.exp(-z))
  return x

vsigmoid = numpy.vectorize(sigmoid)

def loss(h, y):
  if (y==1):
    return -numpy.log(h)
  else:
    return -numpy.log(1-h)
      

vloss = numpy.vectorize(loss)

def cost(x, w, y):
  lin_activation = w.dot(x)
  h = vsigmoid(lin_activation)
  l = vloss(h,y)
  return (float(1)/float(x.shape[1]))*numpy.sum(l)

def grad_descent(x, w, y, alph, lamb):
  lin_activation = w.dot(x)
  h = vsigmoid(lin_activation)
  diff = (h-y)
  der = numpy.transpose(x.dot(numpy.transpose(diff)))
  reg = (lamb)*w #regularized derivative
  reg_der = der+reg
  lr_reg_der = alph*reg_der # regularize derivative with learing rate
  return w-lr_reg_der
 
#initialize variable for training and validation
batch_size = 1
i=0
convergence_const = 0.00005
print("reading t.out")
t = numpy.loadtxt('t.out')
print("reading train_matrix.out")
train_matrix = numpy.loadtxt('train_matrix.out')
print("start training...")
#perform gradient descent with selected lambda
lamb = 0.9
w = numpy.random.uniform(-1,1,train_matrix.shape[0])
converged = False
distance = 0
indecies = numpy.arange(train_matrix.shape[1])
while not converged:
  numpy.random.shuffle(indecies)
  for j in range(0, int(train_matrix.shape[1]/batch_size)):
    i+=1
    #get the batch
    random_dp = indecies[j]
    batch = train_matrix[:, [jj for jj in range(random_dp*batch_size,(random_dp+1)*batch_size)]]
    batch_t = t[[jj for jj in range(random_dp*batch_size,(random_dp+1)*batch_size)]]
    w_new = grad_descent(batch, w, batch_t, 1.1*math.pow(i, -0.9), lamb)
    distance = numpy.linalg.norm(w_new-w)
    w = w_new
  if (distance<=convergence_const):
    converged = True

#Test
correct = 0
total = 0
print("reading vectorizer.out")
vectorizer = pickle.load(open('vectorizer.out', 'rb'))
print("reading transformer.out")
transformer = pickle.load(open('transformer.out', 'rb'))
stops = nltk.corpus.stopwords.words("english")
with open('test.csv', 'rb') as train_file:
  reader = csv.reader(train_file, delimiter=',', quotechar='|')
  for row in reader:
    #get the training set and build training matrix
    if (len(row)>1):
      if (row[0] == "spam"):
        label = 1
      else:
        label = 0

      sentence = row[1]
      #get rid of mess in the text
      filtered_sentence = clean(sentence, stops)
      counts = vectorizer.transform([filtered_sentence]).toarray()
      tfidf = transformer.fit_transform(counts)
      lin_act = tfidf.dot(numpy.transpose(w))
      prediction = numpy.amin(lin_act)
      
      if (label == 1 and prediction>=0):
        correct+=1
      if (label == 0 and prediction<0):
        correct+=1

      total+=1

print("correct={0} total={1} accuracy={2}".format(correct, total, float(correct)/float(total)))
