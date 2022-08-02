from statistics import stdev
import tensorflow as tf
import numpy as np
from preprocess import get_data


class Model(tf.keras.Model):
    def __init__(self, vocab_size):

        """
        The Model class predicts the next words in a sequence,
        Feel free to initialize any variables that you find necessary in the constructor.
        vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # TODO: initialize vocab_size, emnbedding_size 
        self.vocab_size = vocab_size
        self.embedding_size = 50
        self.batch_size = 200 #128
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
        stddev=.1
       
        # TODO: initialize embeddings and forward pass weights (weights, biases)
        self.E = tf.Variable(tf.random.normal(shape=[self.vocab_size, self.embedding_size], stddev=stddev))
        self.W = tf.Variable(tf.random.normal(shape=[2*self.embedding_size, self.vocab_size], stddev=stddev))
        self.b = tf.Variable(tf.random.normal([self.vocab_size], stddev=stddev))


    def call(self,inputs):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        :param inputs: word ids of shape (batch_size, 2)
        :return: prbs: The batch element probabilities as a tensor of shape (batch_size, vocab_size)
        """
        input_1 = inputs[:,0]
        input_2 = inputs[:,1]
        
        embedding_1 = tf.nn.embedding_lookup(self.E, input_1)
        embedding_2 = tf.nn.embedding_lookup(self.E, input_2)
        embedding = tf.concat([embedding_1, embedding_2], axis=1)
        
        logits = tf.matmul(embedding, self.W) + self.b
        return tf.nn.softmax(logits)

    def loss_function(self,logits,labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        :param prbs: a matrix of shape (batch_size, vocab_size)
        :return: the loss of the model as a tensor of size 1
        """
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits))

def train(model, train_input, train_labels):
    """
    Runs through one epoch - all training examples. 
    You should take the train input and shape them into groups of two words.
    Remember to shuffle your inputs and labels - ensure that they are shuffled in the same order. 
    Also you should batch your input and labels here.
    :param model: the initilized model to use for forward and backward pass
    :param train_input: train inputs (all inputs for training) of shape (num_inputs,2)
    :param train_input: train labels (all labels for training) of shape (num_inputs,)
    :return: None
    """
    len_train_i=len(train_input)
    arr=list(range(1,len_train_i))
    shuffle = tf.random.shuffle(arr)
    
    train_input = tf.gather(train_input, shuffle)
    train_labels = tf.gather(train_labels, shuffle)

    total_batches = len(train_input)/model.batch_size
    total_batches=int(total_batches)
    
    for i in range(total_batches):
        slice=i*model.batch_size
        slice_i=(i+1)*model.batch_size
        
        input = train_input[slice:slice_i]
        label = train_labels[slice:slice_i]
 
        with tf.GradientTape() as tape:
            logits = model.call(input)
            loss_function = model.loss_function(logits, label)
        gradients = tape.gradient(loss_function, model.trainable_variables)
        apply_grad=zip(gradients, model.trainable_variables)
        
        model.optimizer.apply_gradients(apply_grad)

def test(model, test_input, test_labels):
    """
    Runs through all test examples. You should take the test input and shape them into groups of two words.
    And test input should be batched here as well.
    :param model: the trained model to use for prediction
    :param test_input: train inputs (all inputs for testing) of shape (num_inputs,2)
    :param test_input: train labels (all labels for testing) of shape (num_inputs,)
    :returns: the perplexity of the test set
    """
    total_batches = len(test_input)/model.batch_size
    total_batches= int(total_batches)
    
    totalloss = 0
   
    for i in range(total_batches):
        slice=i*model.batch_size
        slice_i=(i+1)*model.batch_size

        input = test_input[slice:slice_i]
        label = test_labels[slice:slice_i]
        
        logits = model.call(input)
        
        totalloss = totalloss + model.loss_function(logits, label)
    
    avg_loss = totalloss/total_batches

    perplexity = tf.exp(avg_loss) #https://stackoverflow.com/questions/41881308/how-to-calculate-perplexity-of-rnn-in-tensorflow
    
    return perplexity

def generate_sentence(word1, word2, length, vocab,model):
    """
    Given initial 2 words, print out predicted sentence of target length.
    :param word1: string, first word
    :param word2: string, second word
    :param length: int, desired sentence length
    :param vocab: dictionary, word to id mapping
    :param model: trained trigram model
    """
    reverse_vocab = {idx:word for word, idx in vocab.items()}
    output_string = np.zeros((1,length), dtype=np.int)
    output_string[:,:2] = vocab[word1], vocab[word2]

    for end in range(2,length):
        start = end - 2
        output_string[:, end] = np.argmax(model(output_string[:,start:end]), axis=1)
    text = [reverse_vocab[i] for i in list(output_string[0])]
    
    print(" ".join(text))

def main():
    trainid, testid, vocab = get_data()
    vocab_size = len(vocab)

    train_input = []
    test_input = []
    
    reshape_size=(-1,2)
    
    length_train_id=len(trainid)
    length_test_id=len(testid)
    
    test_label = testid[2:]
    train_label = trainid[2:]
    
    for i in range(length_train_id):
        if i < length_train_id - 2:
            arr = (trainid[i], trainid[i+1])
            train_input.append(arr)
        else:
            pass
  
    train_input = np.reshape(train_input, reshape_size)
    
    for j in range(length_test_id):
        if j < len(testid) - 2:
            arr = (testid[j], testid[j+1])
            test_input.append(arr)
        else:
            pass
    test_input = np.reshape(test_input, reshape_size)
   
    model = Model(vocab_size)
    train(model, train_input, train_label)
    
    perp = test(model, test_input, test_label)
    
    print(perp, 'is the perplexity score')
    
    
if __name__ == '__main__':
    main()