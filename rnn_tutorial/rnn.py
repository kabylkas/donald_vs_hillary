import pandas as pd, numpy as np, tensorflow as tf
import tweets

#initialize the class
t = tweets.tweets("../input/train.csv")

#tokenize the string based on the data in train.csv
for tok in t.tokenize("the war is coming! what you gonna do, bro? don't worry, brasdf!"):
  print(t.d[tok])

