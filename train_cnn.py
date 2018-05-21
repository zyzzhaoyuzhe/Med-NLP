"Training CNN"
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
import datetime
# import data_helpers
from nn_models import TextCNN, TextCNN_field_aware
from tensorflow.contrib import learn
import utils
import data_helpers
import pickle

# Load
df_processed = pickle.load(open('Data/DataFrame_processed.p', 'rb'))

# To predict PAST
TO_PREDICT = 'Past'
FIELDS = [
#     'history',
    'findings',
#     'comparison',
    'impression'
]
df_filtered = df_processed[~df_processed[TO_PREDICT].isnull() & df_processed[TO_PREDICT] != 0].sample(frac=1, random_state=1)
df_filtered = df_filtered[[TO_PREDICT] + FIELDS]

df_train = df_filtered.iloc[:1220]
y_train = np.array(df_train[TO_PREDICT].astype(int))

df_test = df_filtered.iloc[1220:]
y_test = np.array(df_test[TO_PREDICT].astype(int))

print(df_train.shape)
print(df_test.shape)

# ## Training data (Field Unaware)
# x_train_text = utils.Dataframe_Proc.df2text(df_train, df_train.columns[1:])
# word2idx, idx2word = utils.Text_Proc.ngram_vocab_processor(x_train_text, ngram=1, min_count=2)
# x_train = np.array(utils.Text_Proc.encode_texts(x_train_text, word2idx, maxlen=200))
#
# y_train = df_train[TO_PREDICT].values[:, None]
# y_train = np.concatenate([(y_train + 1) / 2, (1 - y_train) / 2], axis=1).astype(np.int)
#
# x_dev_text = utils.Dataframe_Proc.df2text(df_test, df_test.columns[1:])
# x_dev = np.array(utils.Text_Proc.encode_texts(x_dev_text, word2idx, maxlen=x_train.shape[1]))
#
# y_dev = df_test[TO_PREDICT].values[:, None]
# y_dev = np.concatenate([(y_dev + 1) / 2, (1 - y_dev) / 2], axis=1).astype(np.int)
# x_text = utils.Dataframe_Proc.df2text(df_train, df_train.columns[1:])


## Training data (Field Aware)
maxlen = [125, 100]

# get vocab.
x_train_text = utils.Dataframe_Proc.df2text(df_train, df_train.columns[1:])
word2idx, idx2word = utils.Text_Proc.ngram_vocab_processor(x_train_text, ngram=1, min_count=2)
del x_train_text

# Training Set
cache = []
for idx, field in enumerate(df_train.columns[1:]):
    cache.append(np.array(utils.Text_Proc.encode_texts(df_train[field].values, word2idx, maxlen=maxlen[idx])))
x_train = np.concatenate(cache, axis=1)

y_train = df_train[TO_PREDICT].values[:, None]
y_train = np.concatenate([(y_train + 1) / 2, (1 - y_train) / 2], axis=1).astype(np.int)

# Dev Set
cache = []
for idx, field in enumerate(df_train.columns[1:]):
    cache.append(np.array(utils.Text_Proc.encode_texts(df_test[field].values, word2idx, maxlen=maxlen[idx])))
x_dev = np.concatenate(cache, axis=1)

y_dev = df_test[TO_PREDICT].values[:, None]
y_dev = np.concatenate([(y_dev + 1) / 2, (1 - y_dev) / 2], axis=1).astype(np.int)



# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# # Data Preparation
# # ==================================================
#
# # load data
# ordered_names = [u'study',
#                  u'history',
#                  u'comparison',
#                  u'technique',
#                  u'findings',
#                  u'impression',
#                  u'signed by',
#                  ]
#
# filename = 'Data/upto1528.xlsx'
# df_raw = pd.read_excel(open(filename, 'rb'))
#
# # Data is stored in df
# ps = utils.Parser()
# ps.parse(df_raw)
# df = ps.df
# for idx, row in df['findings'].items():
#     try:
#         text, velos = utils.parse_findings(row)
#         df.at[idx, 'findings'] = text
#         for n, v in velos:
#             df.at[0, n] = v
#     except:
#         pass
# discardField = ['Report Text']
# foo = [item for item in df.columns.tolist() if item not in ordered_names+discardField]
# foo.sort()
# CORE_COL = ordered_names + foo
# df = df[CORE_COL]
# df = pd.concat([df_raw[['Past', 'Present', 'Left', 'Right', 'Count']], df[CORE_COL]], axis=1)
# # turn null to []
# df = utils.null2empty(df, ['history', 'impression', 'comparison'])
# print(df.shape)
#
# x_text = utils.df2texts(df, 'findings')
# word2idx, idx2word = utils.ngram_vocab_processor(x_text, ngram=1, min_count=2)
# x = np.array(utils.encode_texts(x_text, word2idx))
#
# y = df['Past'].values[:, None]
# y = np.concatenate([(y + 1) / 2, (1 - y) / 2], axis=1).astype(np.int)
#
# # Randomly shuffle data
# np.random.seed(10)
# shuffle_indices = np.random.permutation(np.arange(len(y)))
# x_shuffled = x[shuffle_indices]
# y_shuffled = y[shuffle_indices]
#
# # Split train/test set
# # TODO: This is very crude, should use cross-validation
# dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
# x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
# y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
#
#
#
#
#
# print("Vocabulary Size: {:d}".format(len(word2idx)))
# print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # cnn = TextCNN(
        #     sequence_length=x_train.shape[1],
        #     num_classes=y_train.shape[1],
        #     vocab_size=len(word2idx),
        #     embedding_size=FLAGS.embedding_dim,
        #     filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
        #     num_filters=FLAGS.num_filters,
        #     l2_reg_lambda=FLAGS.l2_reg_lambda)

        cnn = TextCNN_field_aware(sequence_lengths=maxlen,
                                  num_classes=y_train.shape[1],
                                  vocab_size=len(word2idx),
                                  embedding_size=FLAGS.embedding_dim,
                                  filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                                  num_filters=FLAGS.num_filters,
                                  l2_reg_lambda=FLAGS.l2_reg_lambda)


        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        # vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

