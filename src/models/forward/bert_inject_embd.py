from src.graph_modification.inject import inject_nodes
from src.models.helpers.bert import get_shape_list,reshape_from_matrix,reshape_to_matrix
from src.models.tf_helpers import maybe_print, dense_layer_tf
import tensorflow as tf

def reshape_2D_bert_to_3D(bert_layer):
    with tf.name_scope('reshape_to_3D'):
        graph = tf.get_default_graph()
        tensors = tf.contrib.graph_editor.get_tensors(graph)
        # get original 3D input shape from embeddings (input of transformer
        input_ids = [t for t in tensors if 'bert_lookup_apply_tokens/bert/embeddings/LayerNorm/batchnorm/add_1:0' in t.name][0]
        # subword_cutoff = 200
        # input_ids.set_shape([None, subword_cutoff, 768]) # set number of subword tokens to print shapes during debugging
        input_shape = get_shape_list(input_ids, expected_rank=3) # (batch_size, seq_length, bert_dim)
        # reshape to 3 dim
        bert_layer_3D = reshape_from_matrix(bert_layer, input_shape)
        return bert_layer_3D

def forward_propagation(input_dict, filter_size, filter_number, classes, encoder, sentence_sizes=None, affinity=None, hidden_layer=0,
                        residual_connection=False, similarity_score=False, concat=False, reduction_factor=2, sum_attention=False, dropout=0,
                        skip_pool=False, aux_loss=0, aff_aggr=None, topic_encoder=False, topic_affinity=None, injection_location='embd',
                        seed_list=[], embd_scope='word', scaling_factor=None, scaling_vector=None, gating=False, print_dim=False):
    """
    Implements the forward propagation for the model:
    Gated injection into BERT (GiBERT)

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    logits -- the output of the last LINEAR unit
    """
    if print_dim:
        print('---')
        print('Model: BERT embd inject')
        print('---')

    def embd_injection(bert_layer, W_T, D_T1=None, D_T2=None, embd_layer=True, scaling_factor=None, scaling_vector=None, gating=False):
        '''
        Injects embds by projecting them to BERT dim and adding them to existing embeddings
        :param bert_layer: name of node after which to inject tensor
        :param W_T1: word embeddings of first sentence with (batch, sent_len_1, embd_dim)
        :param W_T2: word embeddings of first sentence with (batch, sent_len_2, embd_dim)
        :return: last injected tensor
        '''

        if embd_layer:
            scope = '/'.join(bert_layer.name.split('/')[:3])
        else:
            scope = '/'.join(bert_layer.name.split('/')[:4])

        with tf.name_scope(scope + '/injection/'):

            ### reshaping BERT input if necessary ###
            if embd_layer:
                # embeddings already come as 3D tensor, so no need to reshape
                bert_layer_3D = bert_layer  # (batch, subword_units, bert_hidden)
            else:
                # reshape flattened 2D matrix to 3D tensor for higher layers
                bert_layer_3D = reshape_2D_bert_to_3D(bert_layer) # (batch, subword_units, bert_hidden)

            input_shape = get_shape_list(bert_layer_3D, expected_rank=3)
            maybe_print([bert_layer_3D],['3D BERT layer'],True)  # (batch, subword_units, bert_hidden)
            bert_dim = input_shape[2]

            ### using word embds only ###
            if D_T1 is None:
                injected = W_T  # (batch, subword_units, embd_dim)

            ### combining word embds with document embds ###
            else:
                # prepare CLS representation based on document embds
                with tf.name_scope('sent_pair_rep'):
                    addition = tf.add(D_T1, D_T2, 'plus')  # (batch,embd_dim)
                    subtraction = tf.subtract(D_T1, D_T2, 'minus')  # (batch,embd_dim)
                    multiplication = tf.multiply(D_T1, D_T2, 'times')  # (batch,embd_dim)
                    all = tf.stack([addition, subtraction, multiplication, D_T1, D_T2], 2)  # concat along new axis (batch,embd_dim,5)
                    maybe_print([all], ['document embd based [cls] representation'], print_dim)
                    # project back to one subword unit
                    cls_rep = dense_layer_tf(all, output_len=1, activation=None, dropout=dropout, name='cls_projection',
                                             share_weights=False, seed_list=seed_list)  # (batch, embd_dim, 1)
                    cls_rep = tf.transpose(cls_rep, [0, 2, 1])  # (batch, 1, embd_dim)
                    maybe_print([cls_rep], ['projected [cls] representation'], print_dim)

                # combine word and document embds
                with tf.name_scope('combine_word_doc'):
                    # bert_rep = bert['sequence_output'][:, 0, :]  # shape (batch, embd_dim)
                    rest_W_T = W_T[:, 1:, :]
                    maybe_print([rest_W_T], ['W_T without [cls]'], print_dim)  # (batch, subword_units-1, embd_dim)
                    injected = tf.concat([cls_rep, rest_W_T], 1)  # (batch, subword_units, embd_dim)

                maybe_print([injected],['injected'],True)  # (batch, subword_units, embd_dim)

            ### projecting embds to bert dimensionality ###
            with tf.name_scope('projection'):
                backprojected_layer = dense_layer_tf(injected, output_len=bert_dim, activation=None,
                                                     dropout=dropout, name='projection', share_weights=False,
                                                     seed_list=seed_list) # (batch, subword_units, bert_hidden)
                maybe_print([backprojected_layer], ['injected information projected to BERT dim'], print_dim) # (batch, subword_units, bert_hidden)

            ### scaling/gating mechanisms ###
            if gating:
                # dynamic gate (method 3)
                combined = tf.concat([backprojected_layer, bert_layer_3D], 2) # (batch, subword_units, 2*bert_hidden)
                gate = dense_layer_tf(combined, output_len=bert_dim, activation='sigmoid',dropout=0,
                                                     name='gate', share_weights=False,
                                                     seed_list=seed_list)  # (batch, subword_units, bert_hidden)
                maybe_print([gate], ['gate'], print_dim)  # (batch, subword_units, bert_hidden)
                gated_injection = tf.multiply(gate,backprojected_layer) # elementwise multiplication
                combined = bert_layer_3D + (tf.Variable(0.0, name='scaling_factor') * gated_injection)
                maybe_print([backprojected_layer], ['gated backprojection'],print_dim)  # (batch, subword_units, bert_hidden)

            else:
                # add activation
                backprojected_layer = tf.tanh(backprojected_layer)

                # scaling with scalar (method 1)
                if not scaling_factor is None:
                    backprojected_layer = tf.Variable(scaling_factor,name='scaling_factor')*backprojected_layer

                # scaling with vector (method 2)
                elif not scaling_vector is None:
                    vector = tf.Variable([scaling_vector] * bert_dim, name='scaling_vector')
                    maybe_print([vector], ['scaling vector'], True)  # (batch, embd_dim)
                    backprojected_layer = tf.multiply(vector, backprojected_layer)

                with tf.name_scope('residual_connection'):
                    combined = backprojected_layer + bert_layer_3D

            ### reshaping final output if necessary ###
            if embd_layer:
                output = combined
            else:
                # reshape output for midlayer injection
                with tf.name_scope('reshape_to_2D'):
                    output = reshape_to_matrix(combined)
            maybe_print([bert_layer,output],['input','output'],print_dim)
            # assert bert_layer.get_shape().as_list() == output.get_shape().as_list()
        return output

    ### prepare inputs ###

    args = {} # collect additional arguments for inject function

    # pretrained word embds
    if 'word+avg_align' in embd_scope:
        # aligned word embds
        args['W_T'] = input_dict['W_T'] # (batch, seq_len, embd_dim)
        # avg word embds to get alternative doc embd (based on separate word embds for s1 and s2)
        args['D_T1'] = tf.reduce_mean(input_dict['W_T1'],axis=1,name='avg_W_T1') # (batch, embd_dim)
        args['D_T2'] = tf.reduce_mean(input_dict['W_T2'],axis=1,name='avg_W_T2') # (batch, embd_dim)
    elif 'align' in embd_scope:
        args['W_T'] = input_dict['W_T'] # (batch, seq_len, embd_dim)
    else:
        raise NotImplementedError()

    for k in args.keys():
        maybe_print([args[k]], [k], print_dim)
    args['scaling_factor'] = scaling_factor
    args['scaling_vector'] = scaling_vector
    args['gating'] = gating

    graph = tf.get_default_graph()

    # determine injection point
    if injection_location == 'embd':
        args['embd_layer'] = True
        if 'word+avg' in embd_scope:
            graph = inject_nodes(graph, 'bert_lookup_apply_tokens/bert/embeddings/add_1:0', embd_injection, args)
        elif 'word' in embd_scope:
            graph = inject_nodes(graph, 'bert_lookup_apply_tokens/bert/embeddings/add_1:0', embd_injection, args)
        else:
            raise NotImplementedError('{} not supported as embd scope'.format(embd_scope))
    else:
        args['embd_layer'] = False
        midlayer_node = 'bert_lookup_apply_tokens/bert/encoder/layer_{}/output/LayerNorm/batchnorm/add_1:0'.format(injection_location)
        if 'word+avg' in embd_scope:
            graph = inject_nodes(graph, midlayer_node, embd_injection, args)
        elif 'word' in embd_scope:
            graph = inject_nodes(graph, midlayer_node, embd_injection, args)
        else:
            raise NotImplementedError('{} not supported as embd scope'.format(embd_scope))

    ### final bert layer representation ###
    bert = input_dict['E1']
    # bert has 2 keys: sequence_output which is output embedding for each token and pooled_output which is output embedding for the entire sequence.
    with tf.name_scope('bert_rep'):
        # pooled output (containing extra dense layer)
        bert_rep = bert['pooled_output'] # pooled output over entire sequence
        bert_rep = tf.layers.dropout(inputs=bert_rep, rate=dropout, seed=seed_list.pop(0)) # (batch, BERT_dim)
        maybe_print([bert_rep], ['pooled BERT'], print_dim)

    # classification layer
    with tf.name_scope('output_layer'):
        hidden_size = bert_rep.shape[-1].value
        output_weights = tf.get_variable(
            "output_weights", [classes, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02, seed=seed_list.pop(0)))
        output_bias = tf.get_variable(
            "output_bias", [classes], initializer=tf.zeros_initializer())
        logits = tf.matmul(bert_rep, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        maybe_print([logits], ['output layer'], print_dim)

    output = {'logits':logits}

    return output