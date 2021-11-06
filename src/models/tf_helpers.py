import tensorflow as tf

def maybe_print(elements, names, test_print):
    if test_print:
        for e, n in zip(elements, names):
            print(n + " shape: " + str(e.get_shape()))


def create_placeholders(sentence_lengths, classes, bicnn=False, sparse=True, bert=False):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    sentence length -- scalar, width of sentence matrix 
    classes -- scalar, number of classes

    Returns:
    X -- placeholder for the data input, of shape [None, sentence_length] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, classes] and dtype "float"
    """
    sentence_length = sentence_lengths[0]
    if sparse:
        Y = tf.placeholder(tf.int64, [None, ], name='labels')
    else:
        Y = tf.placeholder(tf.int32, [None, classes], name='labels')
    if bert:
        # BERT Placeholders # no names!!
        X1 = tf.placeholder(dtype=tf.int32, shape=[None, None]) # input ids
        X1_mask = tf.placeholder(dtype=tf.int32, shape=[None, None]) # input masks
        X1_seg = tf.placeholder(dtype=tf.int32, shape=[None, None]) # segment ids
        return X1,X1_mask,X1_seg,Y
    else:
        X = tf.placeholder(tf.int32, [None, sentence_length], name='XL')
        if bicnn:
            sentence_length2 = sentence_lengths[1]
            X2 = tf.placeholder(tf.int32, [None, sentence_length2], name='XR')
            return X, X2, Y
        else:
            return X, Y

def create_word_topic_placeholders(sentence_lengths):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    :param sentence lengths: scalar, width of sentence matrix 
    :param num_topics: number of topics for topic model
    :param dim: dimensions of word topics, should be 2, 3 or None

    Returns:
    X -- placeholder for the data input, of shape [None, sentence_length] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, classes] and dtype "float"
    """
    T1 = tf.placeholder(tf.int32, [None, sentence_lengths[0]], name='W_TL')
    T2 = tf.placeholder(tf.int32, [None, sentence_lengths[1]], name='W_TR')
    return T1,T2

def create_word_topic_placeholder(sentence_length):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    :param sentence lengths: scalar, width of sentence matrix
    :param num_topics: number of topics for topic model
    :param dim: dimensions of word topics, should be 2, 3 or None

    Returns:
    X -- placeholder for the data input, of shape [None, sentence_length] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, classes] and dtype "float"
    """
    WT = tf.placeholder(tf.int32, [None, sentence_length], name='W_T')
    return WT

def initialise_pretrained_embedding(doc_vocab_size, embedding_dim, embedding_placeholder, name='embedding',trainable=True):
    with tf.name_scope(name):
        if trainable:
            print('init pretrained embds')
            embedding_matrix = tf.Variable(embedding_placeholder, trainable=True, name="W",dtype=tf.float32)
        else:
            W = tf.Variable(tf.constant(0.0, shape=[doc_vocab_size, embedding_dim]), trainable=False, name="W")
            embedding_matrix = W.assign(embedding_placeholder)
    return embedding_matrix

def lookup_embedding(X, embedding_matrix,expand=True,transpose=True,name='embedding_lookup'):
    '''
    Looks up embeddings based on word ids
    :param X: word id matrix with shape (m, sentence_length)
    :param embedding_matrix: embedding matrix with shape (vocab_size, embedding_dim)
    :param expand: add dimension to embedded matrix or not
    :param transpose: switch dimensions of embedding matrix or not
    :param name: name used in TF graph
    :return: embedded_matrix
    '''
    embedded_matrix = tf.nn.embedding_lookup(embedding_matrix, X, name=name) # dim [m, sentence_length, embedding_dim]
    if transpose:
        embedded_matrix = tf.transpose(embedded_matrix, perm=[0, 2, 1]) # dim [m, embedding_dim, sentence_length]
    if expand:
        embedded_matrix = tf.expand_dims(embedded_matrix, -1) # dim [m, embedding_dim, sentence_length, 1]
    return embedded_matrix

def initialize_parameters(layers,emb_dim,filter_width,filter_number,sentence_lengths,attention=None, summary=False, hidden_layer=False,num_classes=2, sim_score=False):
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Arguments:
    h -- filter height (embedding dimension)
    w -- filter width (sentence length)

    Returns:
    parameters -- a dictionary of tensors containing W1
    size of filter (d1,wd0)
    """
    tf.set_random_seed(1)  # so that your "random" numbers match ours
    #[filter_height,filter_width,input_channels,output_channels]
    parameters = {}

    # weights for ABCNN1
    if attention in ['ABCNN1','ABCNN3']:
        # preconvolution attention will be fed into conv with sentence matrix, therefore 2 channels
        input_channels = 2
        parameters['WL'] = tf.get_variable('WL', [sentence_lengths[1], emb_dim],
                                           initializer=tf.contrib.layers.xavier_initializer(seed=1))
        parameters['WR'] = tf.get_variable('WR', [sentence_lengths[0], emb_dim],
                                           initializer=tf.contrib.layers.xavier_initializer(seed=2))
        if summary:
            tf.summary.histogram('weights_left',parameters['WL'])
            tf.summary.histogram('weights_right',parameters['WR'])
    else:
        input_channels = 1

    # weights in convolution filters
    print('Filter shape:')
    if len(filter_width) == 1:
        filter_name = 'F'
        output_channels = filter_number
        # parameters[layer] = initialize_conv_weight(layer,h,w,in_channel,out_channel)
        parameters[filter_name] = tf.get_variable(filter_name, [emb_dim, filter_width[0], input_channels, output_channels],
                        initializer=tf.contrib.layers.xavier_initializer(seed=0))
        print(parameters[filter_name].get_shape())
    else:
        assert filter_number % len(filter_width) == 0,'Number of filters cannot be devided by number of filter widths without remainder. Change settings.'
        filter_num_per_size = int(filter_number / len(filter_width))
        output_channels = filter_num_per_size
        seed = 0
        for width in filter_width:
            filter_name = 'F{}'.format(width)
            # parameters[layer] = initialize_conv_weight(layer,h,w,in_channel,out_channel)
            parameters[filter_name] = tf.get_variable(filter_name, [emb_dim, width, input_channels, output_channels],
                                                initializer=tf.contrib.layers.xavier_initializer(seed=seed))
            print(parameters[filter_name].get_shape())
            seed += 1

    # weights for hidden layer
    output_length = filter_number * 2
    if sim_score:
        output_length = filter_number * 3
    if hidden_layer:
        # with tf.name_scope('hidden_layer'):
        parameters['hidden_weights'] = tf.get_variable('hidden_weights', [output_length, output_length],
                                                initializer=tf.contrib.layers.xavier_initializer(seed=5))
        parameters['hidden_biases'] = tf.get_variable('hidden_biases', output_length,
                                                   initializer=tf.contrib.layers.xavier_initializer(seed=6))

    # weights for softmax layer
    # with tf.name_scope('softmax_layer'):
    parameters['softmax_weights'] = tf.get_variable('softmax_weights', [output_length, num_classes],
                                       initializer=tf.contrib.layers.xavier_initializer(seed=3))
    parameters['softmax_biases'] = tf.get_variable('softmax_biases', num_classes,
                                               initializer=tf.contrib.layers.xavier_initializer(seed=4))
    if summary:
        tf.summary.histogram('weights_conv', parameters[filter_name])
    return parameters


def compute_cost(logits, Y, loss_fn='cross_entropy', name='main_cost'):
    """
    Computes the cost

    Arguments:
    logits -- output of forward propagation (output of the last LINEAR unit of shape (batch, classes)
    Y -- "true" labels vector of shape (batch,)

    Returns:
    cost - Tensor of the cost function
    """
    # multi class classification (binary classification as special case)
    with tf.name_scope(name):
        if loss_fn=='cross_entropy':
            # maybe_print([logits,Y], ['logits','Y'], True)
            cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y),name='cost')
        elif loss_fn=='bert':
            # from https://github.com/google-research/bert/blob/master/run_classifier_with_tfhub.py
            # probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(Y, depth=2, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            cost = tf.reduce_mean(per_example_loss)
        else:
            raise NotImplemented()
    return cost


def dense_layer_tf(X, output_len, activation, name, dropout=0, share_weights=False, seed_list=[4,10]):
    """
    tf.nn.relu(tf.add(tf.matmul(X, weights), biases)) with 3dim X
    :param X: 3dim tensor (batch,input_dim_1,input_dim_2)
    :param weights:weights: 2dim tensor (input_dim,output_dim)
    :param biases: 1dim tensor (output_dim)
    :param name:
    :param activation: activation function to use ['relu',None]
    :param dropout: dropout probablity
    :return: activations (batch,input_dim_2,output_dim)
    """
    with tf.name_scope(name):
        weight_initializer = tf.contrib.layers.xavier_initializer(seed=seed_list.pop(0))
        bias_initializer = tf.zeros_initializer() # tf.contrib.layers.xavier_initializer(seed=5)
        if activation == 'relu':
            activation = tf.nn.relu
        elif activation == 'sigmoid':
            activation = tf.sigmoid
        if share_weights:
            with tf.variable_scope('ffn_weights', reuse=tf.AUTO_REUSE):
                output = tf.contrib.layers.fully_connected(X,output_len,weights_initializer=weight_initializer,biases_initializer=bias_initializer,activation_fn=activation)
        else:
            output = tf.contrib.layers.fully_connected(X,output_len,weights_initializer=weight_initializer,biases_initializer=bias_initializer,activation_fn=activation)
    output = tf.layers.dropout(inputs=output, rate=dropout,seed=seed_list.pop(0))
    return output