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

    return x, res['target'], res['length']


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
  keep_prob = tf.placeholder_with_default(1.0, []) 

  #embedding layer
  embeddings = tf.get_variable('embedding_matrix', [vocab_size, state_size])
  rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

  #rnn
  cell = tf.nn.rnn_cell.GRUCell(state_size)
  init_state = tf.get_variable('init_state', [1, state_size], initializer=tf.constant_initializer(0.0))
  init_state = tf.tile(init_state, [batch_size, 1])
  rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length=seqlen, initial_state=init_state)

  # Add dropout, as the model otherwise quickly overfits
  rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)

  idx = tf.range(batch_size)*tf.shape(rnn_outputs)[1] + (seqlen - 1)
  last_rnn_output = tf.gather(tf.reshape(rnn_outputs, [-1, state_size]), idx)

  # Softmax layer
  with tf.variable_scope('softmax'):
      W = tf.get_variable('W', [state_size, num_classes])
      b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
  logits = tf.matmul(last_rnn_output, W) + b
  preds = tf.nn.softmax(logits)
  correct = tf.equal(tf.cast(tf.argmax(preds,1),tf.int32), y)
  accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

  return {
    'x': x,
    'seqlen': seqlen,
    'y': y,
    'dropout': keep_prob,
    'loss': loss,
    'ts': train_step,
    'preds': preds,
    'accuracy': accuracy
  }

def train_graph(graph, tr, te, batch_size, num_epochs = 30):
  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    step, accuracy = 0, 0
    tr_losses, te_losses = [], []
    current_epoch = 0
    while current_epoch < num_epochs:
      step += 1
      batch = tr.next_batch(batch_size)
    
      feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2], g['dropout']: 0.6}
      accuracy_, _ = sess.run([g['accuracy'], g['ts']], feed_dict=feed)
      accuracy += accuracy_

      if tr.epochs > current_epoch:
        current_epoch += 1
        tr_losses.append(accuracy / step)
        step, accuracy = 0, 0

        #eval test set
        te_epoch = te.epochs
        while te.epochs == te_epoch:
          step += 1
          batch = te.next_batch(batch_size)
          feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2]}
          accuracy_ = sess.run([g['accuracy']], feed_dict=feed)[0]
          accuracy += accuracy_

        te_losses.append(accuracy / step)
        step, accuracy = 0,0
        print("Accuracy after epoch", current_epoch, " - tr:", tr_losses[-1], "- te:", te_losses[-1])

  return tr_losses, te_losses

#load and hanldle data
t = tweets.tweets("../input/train.csv")
train_len, test_len = np.floor(len(t.df)*0.8), np.floor(len(t.df)*0.2)
train, test = t.df.ix[:train_len-1], t.df.ix[train_len:train_len + test_len]
tr = PaddedDataIterator(train)
te = PaddedDataIterator(test)
batch_size = 128
#tensorflow
g = build_graph(vocab_size=len(t.d), state_size=64, batch_size=batch_size, num_classes=2)
tr_losses, te_losses = train_graph(graph=g, tr=tr, te=te, batch_size=batch_size)
