import csv
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy
import math
import random

DEBUG = 2

#helper function
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
  #der = (float(1)/float(x.shape[1]))*der  #derivative--------------------fix
  reg = (lamb)*w #regularized derivative--
  reg_der = der+reg
  lr_reg_der = alph*reg_der # regularize derivative with learing rate
  return w-lr_reg_der
#utility functions
def debug(v, msg):
  if (v<=DEBUG):
    print(msg)


#get the vectorized training data 
debug(1,"Reading training data...")
train_matrix = numpy.loadtxt('./dumps/train_matrix.out')
t = numpy.loadtxt('./dumps/t.out')

#initialize variable for training and validation
batch_size = 1
convergence_const = 0.00003
k = 10 #k-fold validation
fold_size = int(train_matrix.shape[1]/k)
lambdas = []#numpy.arange(0,100, 0.01)
alphas = []
for i in range(10):
  lambdas.append(random.uniform(0.0,5.0))#numpy.arange(0,100, 0.01)
  alphas.append(random.uniform(0.5,2.0))
print(lambdas)
print(alphas)
report_best_accuracy = []
for alpha in alphas:
  for lamb in lambdas:
    fold_accuracy = []
    for fold in range(0, k):
      w = numpy.random.uniform(-1,1,train_matrix.shape[0])
      #generate folded training set
      debug(1, "")
      debug(1, "Folding data for cross validation...")
      training_set = numpy.delete(train_matrix, [i for i in range(fold*fold_size, (fold+1)*fold_size)],1)
      training_t = numpy.delete(t, [i for i in range(fold*fold_size, (fold+1)*fold_size)])
      validation_set = train_matrix[:,[i for i in range(fold*fold_size, (fold+1)*fold_size)]]
      validation_t = t[[i for i in range(fold*fold_size, (fold+1)*fold_size)]]
      #perform gradient descent
      debug(1, "Starting gradient descent...")
      i=0
      converged = False
      while not converged:
        distance = 0
        for j in range(0, int(training_set.shape[1]/batch_size)):
          i+=1
          #get the batch
          batch = training_set[:, [jj for jj in range(j*batch_size,(j+1)*batch_size)]]
          batch_t = training_t[[jj for jj in range(j*batch_size,(j+1)*batch_size)]]
          w_new = grad_descent(batch, w, batch_t, alpha*math.pow(i, -0.9), lamb)
          distance = numpy.linalg.norm(w_new-w)
          debug(3, "Weight distance={0}".format(distance))
          w=w_new
        debug(3, "Iteration {0}, current loss on training={1}".format(i,cost(training_set,w,training_t)))
        if (distance<=convergence_const or i>1000000):
          converged = True
          debug(1, "Conveged...")
      debug(1, "Validating...")
      correct = 0
      for i in range(0,validation_set.shape[1]):
        label = validation_t[i]
        tfidf = validation_set[:,i]
        lin_activation = tfidf.dot(numpy.transpose(w))
        prediction = numpy.amin(lin_activation)
        if (prediction>=0 and label==1):
          correct+=1
        if (prediction<0 and label==0):
          correct+=1
      accuracy = (float(correct)/float(validation_set.shape[1]))
      debug(3, "Accuracy on validation set(k={0})={1:.2f}%".format(fold+1, accuracy*100))
      fold_accuracy.append(accuracy)

    best_acc = numpy.amax(fold_accuracy)
    report_best_accuracy.append(best_acc)
    debug(1, "***Best accuracy={0:.2f}% | alpha={1} | lambda={2}".format(numpy.amax(best_acc*100), alpha, lamb))
    with open("./dumps/results.txt", a) as out:
      out.write("***Best accuracy={0:.2f}% | alpha={1} | lambda={2}\n".format(numpy.amax(best_acc*100), alpha, lamb))
      

output = open('accuracy.txt', 'w')
for acc in report_best_accuracy:
  output.write("{0}\n".format(acc))
