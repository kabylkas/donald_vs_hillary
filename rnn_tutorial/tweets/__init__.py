import csv
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy
import math
import re
import pickle

class tweets:
  def __init__(self, filename, dict_size = 8000):
    self.dict_size = dict_size
    self.t = []
    self.corpus = []
    self.tokenized_corpus = []
    self.d = self.loadTweets(filename)
    #self.tokenize_corpus()

  #helper functions
  def process(self, sentence):
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
      elif 'co/' in temp_word:
        temp_word = ""

      sen_num+=temp_word+" "

    return sen_num

  def loadTweets(self, fileName):
    #read training set
    d = {}
    with open(fileName, 'r') as train_file:
      reader = csv.reader(train_file, delimiter=',')
      for row in reader:
        #get the target value
        if (len(row)>1):
          if (row[0] == "realDonaldTrump"):
            self.t.append(1)
          else:
            self.t.append(0)

          #get the text
          processed_sen = self.process(row[1])
          self.corpus.append(processed_sen)
          for word in processed_sen.split():
            if word not in d:
              d[word] = 1
            else:
              d[word] += 1

    # One of the ways to improve training is to mark
    # chunk of least frequent wordsrds as "unknowns".
    # This will give better results on testing when
    # program encounters words that were not in the 
    # training set. The following several lines:
    # 1) sort the by frequency
    # 2) removes some chunk at the end
    l = []      
    for word, freq in d.items():
      l.append((word, freq))
    l_sorted_by_freq = sorted(l, key=lambda tup: tup[1], reverse = True)
    l_sorted_by_freq = l_sorted_by_freq[:self.dict_size-1]
    d = [word[0] for word in l_sorted_by_freq]
    d.append("**unknown**")
    return sorted(d)

  def tokenize_corpus(self):
    for sentence in self.corpus:
      token = []
      for word in sentence.split():
        if word in self.d:
          token.append(self.d.index(word))
        else:
          token.append(self.d.index("**unknown**"))
      
      self.tokenized_corpus.append(token)
          
  def tokenize(self, sentence): 
    token = []
    sen = self.process(sentence)
    for word in sen.split():
      if word in self.d:
        token.append(self.d.index(word))
      else:
        token.append(self.d.index("**unknown**"))

    return token


