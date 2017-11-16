import pandas as pd, numpy as np, tensorflow as tf
import tweets

t = tweets.tweets("../input/train.csv")
for tok in t.tokenize("the war is coming! what you gonna do, bro? don't worry, bratan!"):
  print(t.d[tok])

