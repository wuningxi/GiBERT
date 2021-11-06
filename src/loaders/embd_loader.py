import os
import numpy as np

def get_embd_path(opt):
    folder = os.path.join(opt['datapath'], 'embeddings')
    if opt.get('injected_embd',None) in ['counter_fitted']:
        return os.path.join(folder,'counter-fitted-vectors.txt')
    elif opt.get('injected_embd',None) in ['dependency']:
        return os.path.join(folder,'deps.words')
    else:
        raise ValueError("opt must contain 'injected_embd' key with 'counter_fitted' or 'dependency'")

def load_embds(opt, add_unk = True, add_special_tokens = True):
    '''
    Reads word embds and dictionary from file
    :param opt: option dictionary
    :return: word2id_dict,id_word_dict,dict,id_word_dict
    '''
    complete_word_topic_dict = {}
    id_word_dict = {}
    word_id_dict = {}
    embd_matrix = []
    if opt.get('injected_embd',None) in ['counter_fitted','dependency']:
        embd_dim = 300
    else:
        raise ValueError("opt must contain 'injected_embd' key with 'counter_fitted' or 'dependency' but was {}".format(opt['injected_embd']))

    unk_embd = opt.get('unk_embd','random')
    cls_embd = opt.get('cls_embd','random')
    sep_embd = opt.get('sep_embd','random')
    unk_embd_std = opt.get('unk_embd_std',0.02)
    embd_file = get_embd_path(opt)
    print("Reading embds from {}".format(embd_file))
    np.random.seed(1)  # fixed random embd for OOV
    wordid = 0
    if add_unk:
        # add line for UNK word topics
        word = '[UNK]'
        if unk_embd=='zero':
            embd_vector = np.array([0.0]*embd_dim)
        elif unk_embd == 'random':
            embd_vector = generate_random_embd(embd_dim,unk_embd_std)
        else:
            raise ValueError('Choose "random" or "zero" UNK embd')
        id_word_dict[wordid] = word
        word_id_dict[word] = wordid
        complete_word_topic_dict[word] = embd_vector
        embd_matrix.append(embd_vector)
        wordid += 1
    if add_special_tokens:
        # add cls token
        word = '[CLS]'
        if cls_embd == 'zero':
            embd_vector = np.array([0.0] * embd_dim)
        elif cls_embd == 'random':
            embd_vector =  generate_random_embd(embd_dim,unk_embd_std)
        id_word_dict[wordid] = word
        word_id_dict[word] = wordid
        complete_word_topic_dict[word] = embd_vector
        embd_matrix.append(embd_vector)
        wordid += 1
        # add sep token
        word = '[SEP]'
        if sep_embd == 'zero':
            embd_vector = np.array([0.0] * embd_dim)
        elif sep_embd == 'random':
            embd_vector =  generate_random_embd(embd_dim,unk_embd_std)
        id_word_dict[wordid] = word
        word_id_dict[word] = wordid
        complete_word_topic_dict[word] = embd_vector
        embd_matrix.append(embd_vector)
        wordid += 1
        # add pad token
        word = '[PAD]'
        embd_vector = np.array([0.0] * embd_dim)
        id_word_dict[wordid] = word
        word_id_dict[word] = wordid
        complete_word_topic_dict[word] = embd_vector
        embd_matrix.append(embd_vector)
        wordid += 1

    # read other word topics
    with open(embd_file, 'r', encoding='utf-8') as infile:
        for i,line in enumerate(infile):
            # print(line)
            ldata = line.rstrip().split(' ') # \xa in string caused trouble
            word = ldata[0]
            id_word_dict[wordid] = word
            word_id_dict[word] = wordid
            embd_vector = np.array([float(s.replace('[','').replace(']','')) for s in ldata[1:]])
            assert len(embd_vector)==embd_dim
            complete_word_topic_dict[word] = embd_vector
            embd_matrix.append(embd_vector)
            wordid += 1
    embd_matrix = np.array(embd_matrix)
    print('word topic embedding dim: {}'.format(embd_matrix.shape))
    assert len(embd_matrix.shape)==2
    return {'id2word':id_word_dict,'word2id':word_id_dict,'dict':complete_word_topic_dict,'matrix':embd_matrix}

def generate_random_embd(dim,std=0.02):
    mean = 0
    return np.random.normal(mean,std,dim)


if __name__ == '__main__':

    # # Example usage
    # substitute_with_random('dependency')

    # load topic predictions for dataset
    opt = {'datapath': 'data/','injected_embd':'glove'}

    counter_fitted_embds = load_embds(opt,add_unk=False,add_special_tokens=False)

    counter = {'max': counter_fitted_embds['matrix'].max(), 'min': counter_fitted_embds['matrix'].min(),
               'mean': counter_fitted_embds['matrix'].mean(), 'std': counter_fitted_embds['matrix'].std()}
    for k in counter.keys():
        print('{} {}'.format(k,counter[k]))

    # counter_fitted_embds['id2word'][0]
    # counter_fitted_embds['matrix'][0]
    # counter_fitted_embds['dict'].get('accident')
    # counter_fitted_embds['matrix'][counter_fitted_embds['word2id']['accident']]