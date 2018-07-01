import tensorflow as tf
import numpy as np

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = tf.placeholder(tf.int32, name='batch_size', shape=[])

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            # Keeping track of l2 regularization loss (optional)
            l2_loss = tf.constant(0.0)
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


class TextCNN_field_aware(object):
    def __init__(self, sequence_lengths, num_classes, vocab_size,
                 embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sum(sequence_lengths)], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = tf.placeholder(tf.int32, name='batch_size', shape=[])

        # Embedding layer for all fields
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedding = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        num_filters_total = 0

        # break input_x into multiple pieces (specified by sequence_lengths)
        h_pool_flats = []
        sequence_cum_lengths = np.cumsum(sequence_lengths)
        for i in range(len(sequence_lengths)):
            if i == 0:
                st = 0
            else:
                st = sequence_cum_lengths[i-1]
            ed = sequence_cum_lengths[i]

            cache, filter_total = self.cnn_one_field(i, st, ed, vocab_size, embedding_size, filter_sizes, num_filters)
            h_pool_flats.append(cache)
            num_filters_total += filter_total

        h_pool_flat = tf.concat(h_pool_flats, 1)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def cnn_one_field(self, id, seq_start, seq_end, vocab_size, embedding_size, filter_sizes, num_filters):
        sequence_length = seq_end - seq_start
        with tf.name_scope("CNN_%d" % id):
            # embedding layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"):

                # Embedding Layer for all fields
                W = self.embedding
                # Embedding Layer for this Field
                # W = tf.Variable(
                #     tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))

                embedded_chars = tf.nn.embedding_lookup(W, self.input_x[:, seq_start:seq_end])
                embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        return h_pool_flat, num_filters_total


class TextRNN(object):
    "RNN model for text classification"
    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, hidden_size, num_layers, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = tf.placeholder(tf.int32, name='batch_size', shape=[])

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

        # LSTM
        def make_cell(hidden_size):
            return tf.contrib.rnn.LSTMBlockCell(hidden_size, forget_bias=0.0)

        cell = [tf.contrib.rnn.DropoutWrapper(make_cell(hidden_size),
                                              output_keep_prob=self.dropout_keep_prob)
                for _ in range(num_layers)]
        cell = tf.contrib.rnn.MultiRNNCell(cell)

        state = cell.zero_state(self.batch_size, tf.float32)

        outputs = []
        with tf.variable_scope('RNN'):
            for time_step in range(sequence_length):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(self.embedded_chars[:, time_step, :], state)
                outputs.append(tf.expand_dims(cell_output, axis=1))
            self.output = tf.concat(outputs, axis=1)

        #
        with tf.name_scope('output'):
            output_agg = tf.reduce_mean(self.output, axis=1)

            W = tf.get_variable('W',
                                shape=[output_agg.shape[1], num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            # Keeping track of l2 regularization loss (optional)
            l2_loss = tf.constant(0.0)
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.logits = tf.nn.xw_plus_b(output_agg, W, b, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


# TODO: Not a good model
class TextRNN_field_aware(object):
    def __init__(
            self, sequence_lengths, num_classes, vocab_size,
            embedding_size, hidden_size, num_layers, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sum(sequence_lengths)], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = tf.placeholder(tf.int32, name='batch_size', shape=[])

        # Embedding Layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedding = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))

        # break input_x input multiple pieces
        outputs = []
        sequence_cum_lengths = np.cumsum(sequence_lengths)
        for i in range(len(sequence_lengths)):
            if i == 0:
                st = 0
            else:
                st = sequence_cum_lengths[i-1]
            ed = sequence_cum_lengths[i]

            output = self.rnn_one_field(i, st, ed,
                                        vocab_size, embedding_size,
                                        hidden_size, num_layers)
            outputs.append(output)

        with tf.name_scope('output'):
            for i in range(len(outputs)):
                outputs[i] = tf.reduce_mean(outputs[i], axis=1, keep_dims=True)

            output = tf.concat(outputs, axis=1)
            output_agg = tf.reduce_mean(output, axis=1)

            W = tf.get_variable('W',
                                shape=[output_agg.shape[1], num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            self.logits = tf.nn.xw_plus_b(output_agg, W, b, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")



    def rnn_one_field(self, id, seq_start, seq_end,
                      vocab_size, embedding_size,
                      hidden_size, num_layers):
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # Embedding (unified)
            W = self.embedding
            # # Embedding (for each field)
            # W = tf.Variable(
            #     tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))

            embedded_chars = tf.nn.embedding_lookup(W, self.input_x[:, seq_start:seq_end])

        # LSTM
        def make_cell(hidden_size):
            return tf.contrib.rnn.LSTMBlockCell(hidden_size, forget_bias=0.0)

        cell = [tf.contrib.rnn.DropoutWrapper(make_cell(hidden_size),
                                              output_keep_prob=self.dropout_keep_prob)
                for _ in range(num_layers)]
        cell = tf.contrib.rnn.MultiRNNCell(cell)

        state = cell.zero_state(self.batch_size, tf.float32)

        outputs = []
        with tf.variable_scope('RNN_%d' % id):
            for time_step in range(seq_end - seq_start):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(embedded_chars[:, time_step, :], state)
                outputs.append(tf.expand_dims(cell_output, axis=1))
            output = tf.concat(outputs, axis=1)
        return output


class TextRNN_attention(object):
    "RNN model for text classification"
    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, hidden_size, num_layers, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = tf.placeholder(tf.int32, name='batch_size', shape=[])

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

        # LSTM
        def make_cell(hidden_size):
            return tf.contrib.rnn.LSTMBlockCell(hidden_size, forget_bias=0.0)

        cell = [tf.contrib.rnn.DropoutWrapper(make_cell(hidden_size),
                                              output_keep_prob=self.dropout_keep_prob)
                for _ in range(num_layers)]
        cell = tf.contrib.rnn.MultiRNNCell(cell)

        state = cell.zero_state(self.batch_size, tf.float32)

        outputs = []
        with tf.variable_scope('RNN'):
            for time_step in range(sequence_length):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(self.embedded_chars[:, time_step, :], state)
                outputs.append(tf.expand_dims(cell_output, axis=1))
            self.output = tf.concat(outputs, axis=1)

        with tf.name_scope('Attention'):
            output_reshape = tf.reshape(self.output, [-1, hidden_size])

            W = tf.Variable(tf.truncated_normal([hidden_size, 1], stddev=0.1), name="W")
            # W = tf.get_variable('W', shape=[hidden_size, 1])

            E = tf.matmul(output_reshape, W)
            E = tf.reshape(E, [-1, sequence_length])
            E = tf.expand_dims(E, 2)

            # attention weights
            self.alpha = tf.nn.softmax(E, dim=1)

        with tf.name_scope('output'):
            output_agg = tf.reduce_sum(tf.multiply(self.output, self.alpha), axis=1)
            # output_agg = tf.reduce_mean(self.output, axis=1)

            W = tf.get_variable('W',
                                shape=[output_agg.shape[1], num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            self.logits = tf.nn.xw_plus_b(output_agg, W, b, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")





