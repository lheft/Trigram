import numpy as np
import tensorflow as tf
from preprocess import get_data
from tensorflow.keras import Model
import os

# ensures that we run only on cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next words in a sequence.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # TODO: initialize embedding_size, batch_size, and any other hyperparameters

        self.vocab_size = vocab_size
        self.window_size = 20
        self.embedding_size = 200 # TODO
        self.batch_size = 200 # TODO
        learning_rate=.01
        stddev=.1
        GRU_units=200
        FF_units=400
        
        # TODO: initialize embeddings and forward pass weights (weights, biases)
        # Note: You can now use tf.keras.layers!
        # - use tf.keras.layers.Dense for feed forward layers
        # - and use tf.keras.layers.GRU or tf.keras.layers.LSTM for your RNN

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.E = tf.Variable(tf.random.normal(shape=[self.vocab_size, self.embedding_size], stddev=stddev))
        
        self.gru = tf.keras.layers.GRU(GRU_units, return_sequences = True, return_state = True)
        
        self.dense1 = tf.keras.layers.Dense(FF_units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.vocab_size, activation='softmax')
    
    def call(self, inputs, initial_state):
        """
        - You must use an embedding layer as the first layer of your network
        (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, a final_state
        (NOTE 1: If you use an LSTM, the final_state will be the last two RNN outputs,
        NOTE 2: We only need to use the initial state during generation)
        using LSTM and only the probabilites as a tensor and a final_state as a tensor when using GRU
        """

        embedding = tf.nn.embedding_lookup(self.E, inputs)
        gru, state = self.gru(embedding)
        logits = self.dense1(gru)
        probs = self.dense2(logits)

        return probs, state

    def loss(self, probabilities, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param probabilities: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the average loss of the model as a tensor of size 1
        """

        # TODO: Fill in
        # We recommend using tf.keras.losses.sparse_categorical_crossentropy

        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probabilities))


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples (remember to batch!)
    Here you will also want to reshape your inputs and labels so that they match
    the inputs and labels shapes passed in the call and loss functions respectively.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    # TODO: Fill in
    p_train=int((len(train_inputs)/model.window_size)) * model.window_size

    train_inputs=train_inputs[:p_train]
    train_labels=train_labels[:p_train]

    reshape=(-1,model.window_size)

    train_inputs=np.reshape(train_inputs, reshape)
    train_labels=np.reshape(train_labels, reshape)



    num_batches = len(train_inputs)/model.batch_size
    num_batches=int(num_batches)

    num_batch_array=range(num_batches)
  
    for i in num_batch_array:
        slice=i*model.batch_size
        slice_i=(i+1)*model.batch_size

        inputs = train_inputs[slice:slice_i]
        label = train_labels[slice:slice_i]
        
        with tf.GradientTape() as tape:
            logits, _ = model.call(inputs, None)
            loss = model.loss(logits, label)
            
            #if i % 100 == 0:
             #   print(tf.exp(loss))
        
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples (remember to batch!)
    Here you will also want to reshape your inputs and labels so that they match
    the inputs and labels shapes passed in the call and loss functions respectively.

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set
    """

    # TODO: Fill in
    # NOTE: Ensure a correct perplexity formula (different from raw loss)
    
    p_test=int((len(test_inputs)/model.window_size)) * model.window_size
    test_inputs=test_inputs[:p_test]
    test_labels=test_labels[:p_test]

    reshape=(-1,model.window_size)

    test_inputs=np.reshape(test_inputs, reshape)
    test_labels=np.reshape(test_labels, reshape)

    
    num_batches = len(test_inputs)/model.batch_size
    num_batches=int(num_batches)

    total_loss = 0
    num_batch_array=range(num_batches)
  
    for i in num_batch_array:
        slice=i*model.batch_size
        slice_i=(i+1)*model.batch_size

        inputs = test_inputs[slice:slice_i]
        label = test_labels[slice:slice_i]
        logit, dummy = model.call(test_inputs, None)
        total_loss= total_loss+ model.loss(logit,label)

    mean=total_loss/num_batch_array
    return tf.exp(mean)

def generate_sentence(word1, length, vocab, model, sample_n=10):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    """

    # NOTE: Feel free to play around with different sample_n values

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits, previous_state = model.call(next_input, previous_state)
        logits = np.array(logits[0, 0, :])
        top_n = np.argsort(logits)[-sample_n:]
        n_logits = np.exp(logits[top_n]) / np.exp(logits[top_n]).sum()
        out_index = np.random.choice(top_n, p=n_logits)

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    print(" ".join(text))


def main():
    trainid, testid, vocab = get_data()
    # TO-DO:  Separate your train and test data into inputs and labels
    
    train_y = trainid[1:]
    train_x = trainid[:-1]
    test_x = testid[:-1]
    test_y = testid[1:]

    model = Model(len(vocab))
    
    inputtrain = []
    labeltrain = []
    input_test = []
    label_test = []

    windows_train = len(train_x)/model.window_size
    windows_train=int(windows_train)

    windows_test = len(test_x)/model.window_size
    windows_test=int(windows_test)

    for i in range(windows_train):
        slice=i*model.window_size
        slice_i=(i+1)*model.window_size
        
        inputtrain.append(train_x[slice:slice_i])
        labeltrain.append(train_y[slice:slice_i])
 
    for i in range(windows_test):
        slice=i*model.window_size
        slice_i=(i+1)*model.window_size

        input_test.append(test_x[slice:slice_i])
        label_test.append(test_y[slice:slice_i])

    #reshape into arrays to do the jig
    train_inputs = np.reshape(inputtrain, (-1, model.window_size))
    train_labels = np.reshape(labeltrain, (-1, model.window_size))

    test_inputs = np.reshape(input_test, (-1, model.window_size))
    test_labels = np.reshape(label_test, (-1, model.window_size))

    # TODO: Set-up the training step
    train(model, train_inputs, train_labels)
    # TODO: Set up the testing steps
    p = test(model, test_inputs, test_labels)
    # Print out perplexity 
    print(p, 'is the perplexity')
    generate_sentence('elephant', 8, vocab, model)
    

if __name__ == "__main__":
    main()
