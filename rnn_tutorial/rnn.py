import pandas as pd, numpy as np, tensorflow as tf
import tweets

class PaddedDataIterator():
  def __init__(self, df):
    self.df = df
    self.size = len(self.df)
    self.epochs = 0
    self.shuffle()

  def shuffle(self):
    self.df = self.df.sample(frac=1).reset_index(drop=True)
    self.cursor = 0

  def next_batch(self, n):
    if self.cursor+n > self.size:
      self.epochs += 1
      self.shuffle()
    res = self.df.ix[self.cursor:self.cursor+n-1]
    self.cursor += n

    # Pad sequences with 0s so they are all the same length
    maxlen = max(res['length'])
    x = np.zeros([n, maxlen], dtype=np.int32)
    for i, x_i in enumerate(x):
      x_i[:res['length'].values[i]] = res['as_numbers'].values[i]

    return x, res['as_numbers'], res['target'], res['length']


def reset_graph():
  if 'sess' in globals() and sess:
    sess.close()
  tf.reset_default_graph()

def build_graph(vocab_size, state_size, batch_size, num_classes):
  reset_graph()

  #placeholders
  x = tf.placeholder(tf.int32, [batch_size, None])
  seqlen = tf.placeholder(tf.int32, [batch_size])
  y = tf.placeholder(tf.int32, [batch_size])
  keep_prob = tf.constant(1.0)

  #embedding layer
  embeddings = tf.get_variable('embedding_matrix', [vocab_size, state_size])
  rnn_inputs = tf.nn.embedding_lookup(embeddings, x)


#initialize the class, get the data

t = tweets.tweets("../input/train.csv")
train, test = t.df.ix[:train_len-1], t.df.ix[train_len:train_len + test_len]

data = PaddedDataIterator(train)



