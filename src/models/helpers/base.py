from src.logs.training_logs import get_git_sha
import numpy as np

def compute_output_size(sentence_lengths, filter_width, filter_number, concat, hidden, skip_pool, reduction_factor=2):
    dim_L = sentence_lengths[0]
    dim_R = sentence_lengths[1]
    cnn_dims = []
    for i, width in enumerate(filter_width):
        filter_name = 'F{}'.format(i + 1)
        # print(filter_name)
        input_channels = filter_number[i]
        print('Predicted output')
        # conv
        dim_L = int(((dim_L - width) / 2) + 1)
        dim_R = int(((dim_R - width) / 2) + 1)
        # print((dim_L, dim_R, filter_number[i]))
        cnn_dims.append(dim_L * dim_R * filter_number[i])
        # maxpool
        if i == 0 or not skip_pool:
            dim_L = int(((dim_L - 2) / 2) + 1)
            dim_R = int(((dim_R - 2) / 2) + 1)
            # print((dim_L, dim_R, filter_number[i]))
            cnn_dims.append(dim_L * dim_R * filter_number[i])
    for i in range(hidden):
        cnn_dims.append(int(cnn_dims[-1] / reduction_factor))
        # reduction_factor = reduction_factor * 0.45
    if concat:
        cnn_dims.append(cnn_dims[-1] + sentence_lengths[0] + sentence_lengths[1])
    # print(cnn_dims)


def add_git_version(opt):
    '''
    Add git SHA to option dictionary to keep track of code version during experiment
    :return: updated opt
    '''
    opt['git'] = get_git_sha()
    return opt

def skip_MAP(X):
    """
    Returns False if X contains 10 * x examples (necessary for computing MAP at 10), True otherwise
    :param X: 
    :return: 
    """
    if X.shape[0] % 10 == 0:
        return False
    else:
        return True

def convert_to_token_array(id2word, word_ids):
    '''
    Map word ids back to tokens and substitute padding token with '' as preparation for ELMO input
    :param id2word: 
    :param word_ids: 
    :return: 
    '''
    max_len = word_ids.shape[1]
    tokens_input = [[id2word[w].replace('<PAD>', '') for w in s] for s in word_ids]
    # tokens_input_padded = [s + [''] * (max_len - len(s)) for s in tokens_input] #
    return np.array(tokens_input)

def extract_data(data_dict, topic_scope, elmo_embd, extra_test, bert_embd):
    '''
    Extracts relevant data from data_dict for base_model
    :param data_dict: dictionary with preprocessed data as provided by load_data()
    :param topic_scope: '','word','doc', or 'word+doc'
    :param extra_test: boolean, evaluate on extra test set (as with PAWS)
    :return: train_dict, dev_dict, test_dict, test_dict_extra
    '''

    D_T1_train, D_T2_train, D_T1_dev, D_T2_dev, D_T1_test, D_T2_test, D_T1_test_extra, D_T2_test_extra = [None] * 8
    W_T1_train, W_T2_train, W_T1_dev, W_T2_dev, W_T1_test, W_T2_test, W_T1_test_extra, W_T2_test_extra = [None] * 8
    W_T_train, W_T_dev, W_T_test, W_T_test_extra, = [None] * 4


    if bert_embd:
        if extra_test:
            X1_train, X1_dev, X1_test, X1_test_extra = data_dict['E1']
            X1_mask_train, X1_mask_dev, X1_mask_test, X1_mask_test_extra = data_dict['E1_mask']
            X1_seg_train, X1_seg_dev, X1_seg_test, X1_seg_test_extra = data_dict['E1_seg']
            Y_train, Y_dev, Y_test, Y_test_extra = data_dict['L']
        else:
            X1_train, X1_dev, X1_test = data_dict['E1']
            X1_mask_train, X1_mask_dev, X1_mask_test = data_dict['E1_mask']
            X1_seg_train, X1_seg_dev, X1_seg_test = data_dict['E1_seg']
            Y_train, Y_dev, Y_test = data_dict['L']

        # assign topics if required
        if 'doc' in topic_scope:
            if extra_test:
                D_T1_train, D_T1_dev, D_T1_test, D_T1_test_extra = data_dict['D_T1']
                D_T2_train, D_T2_dev, D_T2_test, D_T2_test_extra = data_dict['D_T2']
            else:
                D_T1_train, D_T1_dev, D_T1_test = data_dict['D_T1']
                D_T2_train, D_T2_dev, D_T2_test = data_dict['D_T2']
        if 'word' in topic_scope:
            if extra_test:
                if 'avg_align' in topic_scope:
                    W_T_train, W_T_dev, W_T_test, W_T_test_extra = data_dict['W_T']
                    W_T1_train, W_T1_dev, W_T1_test, W_T1_test_extra = data_dict['W_T1']
                    W_T2_train, W_T2_dev, W_T2_test, W_T2_test_extra = data_dict['W_T2']
                elif 'align' in topic_scope:
                    W_T_train, W_T_dev, W_T_test, W_T_test_extra = data_dict['W_T']
                else:
                    W_T1_train, W_T1_dev, W_T1_test, W_T1_test_extra = data_dict['W_T1']
                    W_T2_train, W_T2_dev, W_T2_test, W_T2_test_extra = data_dict['W_T2']
            else:
                if 'avg_align' in topic_scope:
                    W_T_train, W_T_dev, W_T_test = data_dict['W_T']
                    W_T1_train, W_T1_dev, W_T1_test = data_dict['W_T1']
                    W_T2_train, W_T2_dev, W_T2_test = data_dict['W_T2']
                elif 'align' in topic_scope:
                    W_T_train, W_T_dev, W_T_test = data_dict['W_T']
                else:
                    W_T1_train, W_T1_dev, W_T1_test = data_dict['W_T1']
                    W_T2_train, W_T2_dev, W_T2_test = data_dict['W_T2']
            # word_topic_matrix = data['word_topics']['topic_matrix']

        train_dict = {'E1': X1_train, 'E1_mask': X1_mask_train,'E1_seg': X1_seg_train, 'D_T1': D_T1_train, 'D_T2': D_T2_train, 'W_T1': W_T1_train,
                      'W_T2': W_T2_train, 'W_T': W_T_train, 'Y': Y_train}
        dev_dict = {'E1': X1_dev, 'E1_mask': X1_mask_dev,'E1_seg': X1_seg_dev,  'D_T1': D_T1_dev, 'D_T2': D_T2_dev, 'W_T1': W_T1_dev,
                    'W_T2': W_T2_dev, 'W_T': W_T_dev, 'Y': Y_dev}
        test_dict = {'E1': X1_test, 'E1_mask': X1_mask_test,'E1_seg': X1_seg_test, 'D_T1': D_T1_test, 'D_T2': D_T2_test, 'W_T1': W_T1_test,
                     'W_T2': W_T2_test,'W_T': W_T_test, 'Y': Y_test}

        # shapes of E1, E1_mask and E1_seg always need to match
        assert train_dict['E1'].shape == train_dict['E1_mask'].shape == train_dict['E1_seg'].shape
        assert dev_dict['E1'].shape == dev_dict['E1_mask'].shape == dev_dict['E1_seg'].shape
        assert test_dict['E1'].shape == test_dict['E1_mask'].shape == test_dict['E1_seg'].shape

        if extra_test:
            test_dict_extra = {'E1': X1_test_extra, 'E1_mask': X1_mask_test_extra,'E1_seg': X1_seg_test_extra, 'D_T1': D_T1_test_extra, 'D_T2': D_T2_test_extra,
                               'W_T1': W_T1_test_extra, 'W_T2': W_T2_test_extra, 'W_T': W_T_test_extra, 'Y': Y_test_extra}
            assert test_dict_extra['E1'].shape == test_dict_extra['E1_mask'].shape == test_dict_extra['E1_seg'].shape
        else:
            test_dict_extra = None
        if len(train_dict['Y'].shape)>1:
            assert (train_dict['Y'].shape[1] == dev_dict['Y'].shape[1] == test_dict['Y'].shape[1]), \
                'Inconsistent input dimensions of labels'

    else: # non BERT
        if extra_test:
            X1_train, X1_dev, X1_test, X1_test_extra = data_dict['E1']
            X2_train, X2_dev, X2_test, X2_test_extra = data_dict['E2']
            Y_train, Y_dev, Y_test, Y_test_extra = data_dict['L']
        else:
            X1_train, X1_dev, X1_test = data_dict['E1']
            X2_train, X2_dev, X2_test = data_dict['E2']
            Y_train, Y_dev, Y_test = data_dict['L']

        # assign topics if required
        if 'doc' in topic_scope:
            if extra_test:
                D_T1_train, D_T1_dev, D_T1_test, D_T1_test_extra = data_dict['D_T1']
                D_T2_train, D_T2_dev, D_T2_test, D_T2_test_extra = data_dict['D_T2']
            else:
                D_T1_train, D_T1_dev, D_T1_test = data_dict['D_T1']
                D_T2_train, D_T2_dev, D_T2_test = data_dict['D_T2']
        if 'word' in topic_scope:
            if extra_test:
                W_T1_train, W_T1_dev, W_T1_test, W_T1_test_extra = data_dict['W_T1']
                W_T2_train, W_T2_dev, W_T2_test, W_T2_test_extra = data_dict['W_T2']
            else:
                W_T1_train, W_T1_dev, W_T1_test = data_dict['W_T1']
                W_T2_train, W_T2_dev, W_T2_test = data_dict['W_T2']
            # word_topic_matrix = data['word_topics']['topic_matrix']

        train_dict = {'E1': X1_train, 'E2': X2_train, 'D_T1': D_T1_train, 'D_T2': D_T2_train, 'W_T1': W_T1_train,
                      'W_T2': W_T2_train, 'Y': Y_train}
        dev_dict = {'E1': X1_dev, 'E2': X2_dev, 'D_T1': D_T1_dev, 'D_T2': D_T2_dev, 'W_T1': W_T1_dev,
                    'W_T2': W_T2_dev, 'Y': Y_dev}
        test_dict = {'E1': X1_test, 'E2': X2_test, 'D_T1': D_T1_test, 'D_T2': D_T2_test, 'W_T1': W_T1_test,
                     'W_T2': W_T2_test, 'Y': Y_test}
        if extra_test:
            test_dict_extra = {'E1': X1_test_extra, 'E2': X2_test_extra, 'D_T1': D_T1_test_extra, 'D_T2': D_T2_test_extra,
                               'W_T1': W_T1_test_extra, 'W_T2': W_T2_test_extra, 'Y': Y_test_extra}
        else:
            test_dict_extra = None

        # token preprocessing for elmo
        if elmo_embd:
            # map ids back to tokens as elmo input
            train_dict['T1'] = convert_to_token_array(data_dict['id2word'], X1_train)
            train_dict['T2'] = convert_to_token_array(data_dict['id2word'], X2_train)
            dev_dict['T1'] = convert_to_token_array(data_dict['id2word'], X1_dev)
            dev_dict['T2'] = convert_to_token_array(data_dict['id2word'], X2_dev)
            test_dict['T1'] = convert_to_token_array(data_dict['id2word'], X1_test)
            test_dict['T2'] = convert_to_token_array(data_dict['id2word'], X2_test)
            if extra_test:
                test_dict_extra['T1'] = convert_to_token_array(data_dict['id2word'], X1_test_extra)
                test_dict_extra['T2'] = convert_to_token_array(data_dict['id2word'], X2_test_extra)

        assert (train_dict['E1'].shape[1] == dev_dict['E1'].shape[1] == test_dict['E1'].shape[1]), \
            'Inconsistent input dimensions of features from first sentence'
        assert (train_dict['E2'].shape[1] == dev_dict['E2'].shape[1] == test_dict['E2'].shape[1]), \
            'Inconsistent input dimensions of features from second sentence'
        if len(train_dict['Y'].shape)>1:
            assert (train_dict['Y'].shape[1] == dev_dict['Y'].shape[1] == test_dict['Y'].shape[1]), \
                'Inconsistent input dimensions of labels'

    return train_dict, dev_dict, test_dict, test_dict_extra