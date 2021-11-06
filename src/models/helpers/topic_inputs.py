from src.models.tf_helpers import maybe_print
import tensorflow as tf

def combine_inputs(input_dict,print_dim):
    '''
    Combines topic and embedding inputs based on input_dict
    :param input_dict: 
    :param print_dim: 
    :return: {'X1':X1,'X2':X2,'input_dim':input_dim}
    '''
    # word embeddings
    assert not input_dict['E1'] is None
    if input_dict['W_T1'] is None:
        if input_dict['D_T1'] is None:
            # no topic
            raise ValueError('No topic inputs to combine.')
            # return {'X1':input_dict['E1'],'X2':input_dict['E2']}
        else:
            # doc topic
            return combine_doc_topic(input_dict, print_dim)
    else:
        if input_dict['D_T1'] is None:
            # word topic
            return combine_word_topic(input_dict, print_dim)
        else:
            # word+doc topic
            return combine_word_doc_topic(input_dict, print_dim)

def combine_doc_topic(input_dict, print_dim):
    '''
    Copy document topics for each word in sentence (as ablation) to match dimension for attention matrix
    :param input_dict: 
    :param print_dim: 
    :return: 
    '''
    E1 = input_dict['E1'] # (batch, sent_len_1, embd)
    E2 = input_dict['E2'] # (batch, sent_len_2, embd)
    D_T1 = input_dict['D_T1'] # (batch, num_topics)
    D_T2 = input_dict['D_T2'] # (batch, num_topics)

    maybe_print([D_T1, D_T2], ['TL', 'TR'], print_dim)
    [_, sent_len_l, _] = E1.get_shape().as_list()  # (batch,emb,input_len)
    [_, sent_len_r, _] = E2.get_shape().as_list()  # (batch,emb,input_len)
    [_, topic_num] = D_T1.get_shape().as_list()

    # Copy document topics for each word and combine with word embedding before encoding
    with tf.name_scope('input_L'):
        D_T1 = tf.expand_dims(D_T1, 1) # (batch, 1, num_topics)
        X1 = tf.tile(D_T1, [1, sent_len_l, 1])  # (batch, sent_len_1, num_topics)
    with tf.name_scope('input_R'):
        D_T2 = tf.expand_dims(D_T2, 1) # (batch, 1, num_topics)
        X2 = tf.tile(D_T2, [1,sent_len_r, 1]) # (batch, sent_len_2, num_topics)
    maybe_print([D_T1, D_T2], ['TL', 'TR'], print_dim)
    input_dim = topic_num
    return {'X1':X1,'X2':X2,'input_dim':input_dim}

def combine_word_topic(input_dict, print_dim):
    '''
    Use word topics as are
    :param input_dict: 
    :param print_dim: 
    :return: 
    '''
    X1 = input_dict['W_T1'] # (batch, sent_len_1, num_topics)
    X2 = input_dict['W_T2'] # (batch, sent_len_2, num_topics)

    maybe_print([X1, X2], ['TL', 'TR'], print_dim)
    [_, _, topic_num] = X1.get_shape().as_list()

    input_dim = topic_num
    return {'X1':X1,'X2':X2,'input_dim':input_dim}

def combine_word_doc_topic(input_dict, print_dim):
    '''
    Combine word and document topics
    :param input_dict: 
    :param print_dim: 
    :return: 
    '''
    # word topic model distributions
    W_T1 = input_dict['W_T1'] # (batch, sent_len_1, num_topics)
    W_T2 = input_dict['W_T2'] # (batch, sent_len_2, num_topics)
    # document topic model distributions
    D_T1 = input_dict['D_T1'] # (batch, num_topics)
    D_T2 = input_dict['D_T2'] # (batch, num_topics)

    maybe_print([W_T1, W_T2], ['W_TL', 'W_TR'], print_dim)
    maybe_print([D_T1, D_T2], ['D_TL', 'D_TR'], print_dim)

    [_, _, topic_num] = W_T1.get_shape().as_list()

    # Element-wise multiplication of word and doc topics, then concatenate topics before encoding
    with tf.name_scope('input_L'):
        D_T1 = tf.expand_dims(D_T1, 1) # (batch, 1, num_topics)
        X1 = tf.multiply(W_T1, D_T1)  # (batch, sent_len_1, num_topics)
    with tf.name_scope('input_R'):
        D_T2 = tf.expand_dims(D_T2, 1) # (batch, 1, num_topics)
        X2 = tf.multiply(W_T2, D_T2) # (batch, sent_len_1, num_topics)

    input_dim = topic_num
    return {'X1':X1,'X2':X2,'input_dim':input_dim}