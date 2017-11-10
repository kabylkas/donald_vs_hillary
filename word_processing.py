import csv
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy
import math
import re
import pickle

dictionary = []

#helper functions
def process(sentence):
  count=0
  sen = sentence.lower()

  sen = sen.replace("don't", " do not ")
  sen = sen.replace("won't", " will not ")
  sen = sen.replace("doesn't", " does not ")
  sen = sen.replace("haven't", " have not ")
  sen = sen.replace("hasn't", " has not ")
  sen = sen.replace("can't", " can not ")
  sen = sen.replace("couldn't", " could not ")
  sen = sen.replace("wouldn't", " would not ")

  sen = sen.replace(".", " **period** ")
  sen = sen.replace("—", " **emdash** ")
  sen = sen.replace("–", " **emdash** ")
  sen = sen.replace("-", " **dash** ")
  sen = sen.replace(",", " **comma** ")
  sen = sen.replace("!", " **exclamation** ")
  sen = sen.replace("¡", "")
  sen = sen.replace("?", " **question** ")
  sen = sen.replace("¿", "")
  sen = sen.replace(":", " **colon** ")
  sen = sen.replace(";", " **semicolon** ")
  sen = sen.replace("\"", " **quote** ")
  sen = sen.replace("'s", " **possess** ")
  sen = sen.replace("'", " **singlequote** ")
  sen = sen.replace("‘", " **singlequote** ")
  sen = sen.replace("’", " **singlequote** ")
  sen = sen.replace("“", " **quoteopen** ")
  sen = sen.replace("“", " **quoteclose** ")
  sen = sen.replace("(", " **bracketsopen** ")
  sen = sen.replace(")", " **bracketsclose** ")
  sen = sen.replace("[", " **sqbracketsopen** ")
  sen = sen.replace("]", " **sqbracketsclose** ")
  sen = sen.replace("{", " **curbracketsopen** ")
  sen = sen.replace("}", " **curbracketsclose** ")
  sen = sen.replace("}", " **pipe** ")
  #TODO
  #do for links **link**
  #do for emails **email**
  #maybe do quoteopen quoteclose 
  #maybe do for money **money**
  #dates

  sen_num = ""
  for word in sen.split():
    temp_word = word.replace(" ", "")

    if temp_word.isdigit():
      temp_word = "**number**"

    sen_num+=word+" "

    if word not in dictionary:
      dictionary.append(word)


  return sen_num

#read training set
t = []
corpus = []

with open('./input/train.csv', 'r') as train_file:
  reader = csv.reader(train_file, delimiter=',', quotechar='|')
  for row in reader:
    #get the target value
    if (len(row)>1):
      if (row[0] == "realDonaldTrump"):
        t.append(1)
      else:
        t.append(0)

      #get the text 
      corpus.append(process(row[1]))

dictionary = sorted(dictionary)

