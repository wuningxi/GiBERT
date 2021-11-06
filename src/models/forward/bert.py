from src.models.tf_helpers import maybe_print

def forward_propagation(input_dict, filter_size, filter_number, classes, encoder, sentence_sizes=None, affinity=None, hidden_layer=0,
                        residual_connection=False, similarity_score=False, concat=False, reduction_factor=2, sum_attention=False, dropout=0,
                        skip_pool=False, aux_loss=0, aff_aggr=None, topic_encoder=False, topic_affinity=None, injection_location=None,
                        seed_list=[], topic_scope=None, scaling_factor=None, scaling_vector=None, gating=False, print_dim=False):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    if print_dim:
        print('---')
        print('Model: BERT')
        print('---')

    bert = input_dict['E1']
    # bert has 2 keys: sequence_output which is output embedding for each token and pooled_output which is output embedding for the entire sequence.

    with tf.name_scope('bert_rep'):
        # pooled output (containing extra dense layer)
        bert_rep = bert['pooled_output'] # pooled output over entire sequence
        maybe_print([bert_rep], ['pooled BERT'], print_dim)

        # C vector from last layer corresponding to CLS token
        # bert_rep = bert['sequence_output'][:, 0, :]  # shape (batch, BERT_hidden)
        # maybe_print([bert_rep], ['BERT C vector'], print_dim)

        # add dropout
        bert_rep = tf.layers.dropout(inputs=bert_rep, rate=dropout, seed=seed_list.pop(0))

    if hidden_layer>0:
        raise ValueError("BERT baseline doesn't use additional hidden layers.")

    with tf.name_scope('output_layer'):
        hidden_size = bert_rep.shape[-1].value
        output_weights = tf.get_variable(
            "output_weights", [classes, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02, seed=seed_list.pop(0)))
        output_bias = tf.get_variable(
            "output_bias", [classes], initializer=tf.zeros_initializer())
        logits = tf.matmul(bert_rep, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

    # Softmax Layer
    # output_layer = dense_layer_tf(bert_rep, classes, None, None, name='softmax_layer')
    maybe_print([logits], ['output layer'], print_dim)

    output = {'logits':logits}

    return output