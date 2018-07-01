"Training CNN"
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
import datetime
# import data_helpers
from nn_models import TextCNN, TextCNN_field_aware, TextRNN, TextRNN_field_aware, TextRNN_attention
from tensorflow.contrib import learn
import utils
import data_helpers
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load
df_processed = pickle.load(open('Data/DataFrame_processed.p', 'rb'))

# To predict PAST
TO_PREDICT = 'Grade'
FIELDS = [
    'history',
    'findings',
    'comparison',
    'impression'
]

# df_filtered = df_processed[~df_processed[TO_PREDICT].isnull() & (df_processed[TO_PREDICT] != 0)].sample(frac=1, random_state=1)
df_filtered = df_processed[~df_processed[TO_PREDICT].isnull()].sample(frac=1, random_state=1)
df_filtered = df_filtered[[TO_PREDICT] + FIELDS]

df_train = df_filtered.iloc[:1220]
y_train = np.array(df_train[TO_PREDICT].astype(int))
enc = LabelEncoder()
enc.fit(y_train)
y_train = enc.transform(y_train)

df_test = df_filtered.iloc[1220:]
y_test = np.array(df_test[TO_PREDICT].astype(int))
y_test = enc.transform(y_test)

print(df_train.shape)
print(df_test.shape)

#### Data
maxlen = [100,
          125,
          50,
          100]


# Training data (Field Unaware)
x_train_text = utils.Dataframe_Proc.df2text(df_train, df_train.columns[1:])
word2idx, idx2word = utils.Text_Proc.ngram_vocab_processor(x_train_text, ngram=1, min_count=2)
x_train = np.array(utils.Text_Proc.encode_texts(x_train_text, word2idx, maxlen=sum(maxlen)))

enc = OneHotEncoder(sparse=False)
y_train = enc.fit_transform(y_train[:, None])

x_dev_text = utils.Dataframe_Proc.df2text(df_test, df_test.columns[1:])
x_dev = np.array(utils.Text_Proc.encode_texts(x_dev_text, word2idx, maxlen=x_train.shape[1]))

y_dev = enc.transform(y_test[:, None])

# x_text = utils.Dataframe_Proc.df2text(df_train, df_train.columns[1:])


# Training data (Field Aware)
# get vocab.
x_train_text = utils.Dataframe_Proc.df2text(df_train, df_train.columns[1:])
word2idx, idx2word = utils.Text_Proc.ngram_vocab_processor(x_train_text, ngram=1, min_count=2)
del x_train_text

# # Training Set
# cache = []
# for idx, field in enumerate(df_train.columns[1:]):
#     cache.append(np.array(utils.Text_Proc.encode_texts(df_train[field].values, word2idx, maxlen=maxlen[idx])))
# x_train = np.concatenate(cache, axis=1)
#
# y_train = df_train[TO_PREDICT].values[:, None]
# y_train = np.concatenate([(y_train + 1) / 2, (1 - y_train) / 2], axis=1).astype(np.int)
#
# # Dev
# cache = []
# for idx, field in enumerate(df_train.columns[1:]):
#     cache.append(np.array(utils.Text_Proc.encode_texts(df_test[field].values, word2idx, maxlen=maxlen[idx])))
# x_dev = np.concatenate(cache, axis=1)
#
# y_dev = df_test[TO_PREDICT].values[:, None]
# y_dev = np.concatenate([(y_dev + 1) / 2, (1 - y_dev) / 2], axis=1).astype(np.int)



# Parameters
# ==================================================

###### Model Hyperparameters
# Both CNN and RNN
tf.flags.DEFINE_integer("embedding_dim", 64, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# CNN parameter
tf.flags.DEFINE_string("filter_sizes", "3,4", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 128)")

# RNN parameter
tf.flags.DEFINE_integer('hidden_size', 64, 'Hidden size of LSTM')
tf.flags.DEFINE_integer('num_layers', 1, 'Number of LSTM layers')

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


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # model = TextCNN(
        #     sequence_length=x_train.shape[1],
        #     num_classes=y_train.shape[1],
        #     vocab_size=len(word2idx),
        #     embedding_size=FLAGS.embedding_dim,
        #     filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
        #     num_filters=FLAGS.num_filters,
        #     l2_reg_lambda=FLAGS.l2_reg_lambda)

        # model = TextCNN_field_aware(sequence_lengths=maxlen,
        #                             num_classes=y_train.shape[1],
        #                             vocab_size=len(word2idx),
        #                             embedding_size=FLAGS.embedding_dim,
        #                             filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
        #                             num_filters=FLAGS.num_filters,
        #                             l2_reg_lambda=FLAGS.l2_reg_lambda)

        # model = TextRNN(sequence_length=sum(maxlen),
        #                 num_classes=y_train.shape[1],
        #                 vocab_size=len(word2idx),
        #                 embedding_size=FLAGS.embedding_dim,
        #                 hidden_size=FLAGS.hidden_size,
        #                 num_layers=FLAGS.num_layers,
        #                 l2_reg_lambda=FLAGS.l2_reg_lambda)

        # model = TextRNN_field_aware(sequence_lengths=maxlen,
        #                             num_classes=y_train.shape[1],
        #                             vocab_size=len(word2idx),
        #                             embedding_size=FLAGS.embedding_dim,
        #                             hidden_size=FLAGS.hidden_size,
        #                             num_layers=FLAGS.num_layers,
        #                             l2_reg_lambda=FLAGS.l2_reg_lambda)

        model = TextRNN_attention(sequence_length=sum(maxlen),
                        num_classes=y_train.shape[1],
                        vocab_size=len(word2idx),
                        embedding_size=FLAGS.embedding_dim,
                        hidden_size=FLAGS.hidden_size,
                        num_layers=FLAGS.num_layers,
                        l2_reg_lambda=FLAGS.l2_reg_lambda)


        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # optimizer = tf.train.GradientDescentOptimizer(1e-3)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(model.loss)
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

        if 'CNN' in model.__class__.__name__:
            name = "CNN_{}_{}_".format(FLAGS.embedding_dim, FLAGS.num_filters)
        elif 'RNN' in model.__class__.__name__:
            name = "RNN_{}_{}_{}_".format(FLAGS.embedding_dim, FLAGS.hidden_size, FLAGS.num_layers)


        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", name + timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", model.loss)
        acc_summary = tf.summary.scalar("accuracy", model.accuracy)

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
                model.input_x: x_batch,
                model.input_y: y_batch,
                model.dropout_keep_prob: FLAGS.dropout_keep_prob,
                model.batch_size: len(x_batch)
            }

            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                model.input_x: x_batch,
                model.input_y: y_batch,
                model.dropout_keep_prob: 1.0,
                model.batch_size: len(x_batch)
            }

            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, model.loss, model.accuracy],
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

        #
        feed_dict = {
            model.input_x: x_dev,
            model.input_y: y_dev,
            model.dropout_keep_prob: 1.0,
            model.batch_size: len(x_dev)
        }
        y_pred = sess.run(model.predictions, feed_dict=feed_dict)


print(utils.my_classification_report(np.argmax(y_dev, axis=1),
                                     y_pred))