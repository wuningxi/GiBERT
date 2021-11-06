import importlib
import os
import pickle

import numpy as np

from src.loaders.Quora.build import build

from src.loaders.augment_data import create_large_train
from src.preprocessing.Preprocessor import Preprocessor, get_onehot_encoding, reduce_embd_id_len
from src.loaders.embd_loader import load_embds


def compute_mapping_rates(W_T,mapping_type='unk'):
    if mapping_type == 'unk':
        mapping_rates = []
        for subset in W_T:
            subset_rates = []
            for s in subset:
                non_unk_tokens = len([w for w in s if w not in [0, 1, 2, 3]])
                total_tokens = len([w for w in s if w not in [1, 2, 3]])
                subset_rates.append(non_unk_tokens / total_tokens)
            mapping_rates.append(np.array(subset_rates))
    elif mapping_type == 'sep':
        mapping_rates = []
        for subset in W_T:
            subset_rates = []
            for s in subset:
                non_unk_tokens = len([w for w in s if w == 3])
                total_tokens = len(s)
                subset_rates.append(non_unk_tokens / total_tokens)
            mapping_rates.append(np.array(subset_rates))
    return mapping_rates

def get_filenames(opt):
    filenames = []
    for s in opt['subsets']:
        for t in opt['tasks']:
            prefix = ''
            if opt['dataset'] == 'Quora':
                if s.startswith('p_'):
                    prefix = ''
                else:
                    prefix = 'q_'
            if opt['dataset'] == 'GlueQuora':
                prefix = 'g_'
            if opt['dataset'] == 'PAWS':
                prefix = 'p_'
            if opt['dataset'] == 'MSRP':
                prefix = 'm_'
            if opt['dataset'] == 'STS':
                prefix = 's_'
            filenames.append(prefix+s+'_'+t)
    return filenames

def get_filepath(opt):
    filepaths = []
    for name in get_filenames(opt):
        if 'quora' in name:
            filepaths.append(os.path.join(opt['datapath'], 'Quora', name + '.txt'))
            print('quora in filename')
        else:
            filepaths.append(os.path.join(opt['datapath'], opt['dataset'], name + '.txt'))
    return filepaths

def load_file(filename,onehot=True):
    """
    Reads file and returns tuple of (ID1, ID2, D1, D2, L) if ids=False
    """
    ID1 = []
    ID2 = []
    D1 = []
    D2 = []
    L = []
    with open(filename,'r',encoding='utf-8') as read:
        for i,line in enumerate(read):
            if not len(line.split('\t'))==5:
                print(line.split('\t'))
            id1, id2, d1, d2, label = line.rstrip().split('\t')
            ID1.append(id1)
            ID2.append(id2)
            D1.append(d1)
            D2.append(d2)
            if 's_' in filename:
                if float(label)>=4:
                    label = 1
                elif float(label)<4:
                    label = 0
                else:
                    ValueError()
            L.append(int(label))
    L = np.array(L)
    # L = L.reshape(len(D1),1)
    if onehot:
        classes = L.shape[1] + 1
        L = get_onehot_encoding(L)
        print('Encoding labels as one hot vector.')
    return (ID1, ID2, D1, D2, L)

def get_dataset_max_length(opt):
    '''
    Determine maximum number of tokens in both sentences, as well as highes max length for current task
    :param opt:
    :return: [maximum length of sentence in tokens,should first sentence be shortened?]
    '''
    tasks = opt['tasks']
    if opt['dataset'] in ['Quora','PAWS','GlueQuora']:
        cutoff = opt.get('max_length', 24)
        if cutoff == 'minimum':
            cutoff = 24
        s1_len, s2_len = cutoff, cutoff
    elif opt['dataset']=='MSRP':
        cutoff = opt.get('max_length', 40)
        if cutoff == 'minimum':
            cutoff = 40
        s1_len, s2_len = cutoff, cutoff
    elif 'B' in tasks:
        cutoff = opt.get('max_length', 100)
        if cutoff == 'minimum':
            cutoff = 100
        s1_len, s2_len = cutoff, cutoff
    elif 'A' in tasks or 'C' in tasks:
        cutoff = opt.get('max_length', 200)
        if cutoff == 'minimum':
            s1_len = 100
            s2_len = 200
        else:
            s1_len, s2_len = cutoff,cutoff
    return s1_len,s2_len,max([s1_len,s2_len])

def shuffle_train_examples(input_list):
    '''
    shuffle the training partition of the data
    :param input_list: [ID1,ID2,R1,R2,T1,T2,E1,E1_mask,E1_seg,E2,L1,L2,L]
    :return: input_list with shuffled train portion
    '''
    # input_list=[data_dict['ID1'],data_dict['ID2'],data_dict['R1'],data_dict['R2'],data_dict['T1'],data_dict['T2'],data_dict['E1'],data_dict['E1_mask'],data_dict['E1_seg'],data_dict['E2'],data_dict['L1'],data_dict['L2'],data_dict['L']]
    np.random.seed(1)
    output_list = []
    m = None
    for matrices in input_list:
        if matrices is None or matrices==[]:
            output_list.append(matrices)
        else:
            if m is None:
                # get length and permutation based on first input
                m = len(matrices[0])  # number of examples in train
                permutation = list(np.random.permutation(m)) # get random permutation
            assert len(matrices[0]) == m # ensure same number of examples across all train matrices/lists
            shuffled_train = []
            # shuffle training portion
            for i in permutation:
                shuffled_train += [matrices[0][i]]
            if type(matrices[0])==np.ndarray:
                shuffled_train = np.array(shuffled_train)
            # sanity checks
            assert type(shuffled_train)==type(matrices[0])
            assert len(shuffled_train)==len(matrices[0])
            if type(shuffled_train) is np.ndarray: # numpy array
                assert shuffled_train.shape == matrices[0].shape
                assert set([' '.join(str(e)) for e in shuffled_train]) == set([' '.join(str(e)) for e in matrices[0]]) # for nested lists
            elif type(shuffled_train[0]) is list: # nested list
                assert set([' '.join(e) for e in shuffled_train]) == set([' '.join(e) for e in matrices[0]])
            elif type (shuffled_train) is list: # normal list
                assert set(shuffled_train)==set(matrices[0])
            else:
                print(shuffled_train)
                print(type(shuffled_train) is np.ndarray)
                raise NotImplementedError('Shuffling for {} not supported.'.format(type(shuffled_train)))
            # used shuffled version
            matrices[0] = shuffled_train
            output_list.append(matrices)
    assert len(input_list)==len(output_list)
    return output_list

def reduce_examples(matrices, m):
    '''
    Reduces the size of matrices
    :param matrices:
    :param m: maximum number of examples
    :return:
    '''
    return [matrix[:m] for matrix in matrices]

def create_missing_datafiles(opt,datafile,datapath):
    if not os.path.exists(datapath) and 'large' in datafile:
        create_large_train()
    if not os.path.exists(datapath) and 'quora' in datafile:
        quora_opt = opt
        quora_opt['dataset'] = 'Quora'
        build(quora_opt)

def get_cache_folder(opt):
    return opt['datapath'] + 'cache/'


def load_cache_or_process(opt, cache, onehot, vocab):
    ID1 = []
    ID2 = []
    R1 = []
    R2 = []
    T1 = []
    T2 = []
    L1 = []
    L2 = []
    L = []
    replace_ngrams = opt.get('n_gram_embd',False)
    lemmatize = opt.get('lemmatize',False)
    assert not (replace_ngrams and lemmatize)
    filenames = get_filenames(opt)
    print(filenames)
    filepaths = get_filepath(opt)
    print(filepaths)
    for datafile,datapath in zip(filenames,filepaths):
        create_missing_datafiles(opt,datafile,datapath) # if necessary
        cache_folder = get_cache_folder(opt)
        if not os.path.exists(cache_folder):
            os.mkdir(cache_folder)
        # separate cache for n-gram / no n-gram replacement
        if lemmatize:
            suffix = '_lemma'
        elif replace_ngrams:
            suffix = '_ngram'
        else:
            suffix = ''
        cached_path = cache_folder + datafile + suffix + '.pickle'
        # load preprocessed cache
        print(cached_path)
        if cache and os.path.isfile(cached_path):
            print("Loading cached input for " + datafile + suffix)
            try:
                with open(cached_path, 'rb') as f:
                    if lemmatize:
                        id1, id2, r1, r2, l1, l2, l = pickle.load(f)
                    else:
                        id1, id2, r1, r2, t1, t2, l = pickle.load(f)
            except ValueError:
                Warning('No ids loaded from cache: {}.'.format(cached_path))
                with open(cached_path, 'rb') as f:
                    r1, r2, l = pickle.load(f)
                    id1 = None
                    id2 = None

        # do preprocessing if cache not available
        else:
            print('Creating cache...')
            load_ids = opt.get('load_ids',True)
            if not load_ids:
                DeprecationWarning('Load_ids is deprecated setting. Now loaded automatically.')
            id1, id2, r1, r2, l = load_file(datapath,onehot)
            t1 = Preprocessor.basic_pipeline(r1)
            t2 = Preprocessor.basic_pipeline(r2)
            if lemmatize:
                l1 = Preprocessor.lemmatizer_pipeline(r1)
                l2 = Preprocessor.lemmatizer_pipeline(r2)
            if cache: # don't overwrite existing data if cache=False
                if lemmatize:
                    pickle.dump((id1, id2, r1, r2, l1, l2, l), open(cached_path, "wb")) # Store the new data as the current cache
                else:
                    pickle.dump((id1, id2, r1, r2, t1, t2, l), open(cached_path, "wb")) # Store the new data as the current cache
        ID1.append(id1)
        ID2.append(id2)
        R1.append(r1)
        R2.append(r2)
        L.append(l)
        if lemmatize:
            L1.append(l1)
            L2.append(l2)
        else:
            L1 = None
            L2 = None
            T1.append(t1)
            T2.append(t2)
    return {'ID1': ID1, 'ID2': ID2, 'R1': R1, 'R2': R2,'T1': T1, 'T2': T2, 'L1': L1, 'L2': L2,'L': L}


def load_data(opt,cache=True,numerical=True,onehot=False, write_vocab=False):
    """
    Reads data and does preprocessing based on options file and returns a data dictionary.
    Tokens will always be loaded, other keys depend on settings and will contain None if not available.
    :param opt: option dictionary, containing task and dataset info
    :param numerical: map tokens to embedding ids or not
    :param onehot: load labels as one hot representation or not
    :param write_vocab: write vocabulary to file or not
    :param cache: try to use cached preprocessed data or not
    :return:
        { # essential:
        'ID1': ID1, 'ID2': ID2, # doc ids
        'R1': R1, 'R2': R2, # raw text
        'L': L, # labels
          # optional for word embds:
        'E1': E1, 'E2': E2, # embedding ids
        'embd': embd, # word embedding matrix
        'mapping_rates': mapping_rates,
          # optional for topics:
        'D_T1':D_T1,'D_T2':D_T2, # document topics
        'word_topics':word_topics, # word topic matrix
        'topic_keys':topic_key_table} # key word explanation for topics
    """
    E1 = None
    E1_mask = None
    E1_seg = None
    E2 = None
    L1 = None
    L2 = None
    D_T1 = None
    D_T2 = None
    W_T1 = None
    W_T2 = None
    W_T = None
    topic_key_table = None
    mapping_rates = None
    embd = None
    word_topics = None
    vocab = []
    word2id_bert = None
    id2word_bert = None

    # get options
    shuffle_train = opt.get('shuffle_train',False)
    dataset = opt['dataset']
    module_name = "src.loaders.{}.build".format(dataset)
    my_module = importlib.import_module(module_name)
    my_module.build(opt) # download and reformat if not existing
    topic_scope = opt.get('topic','')
    if not  topic_scope=='':
        topic_type = opt['topic_type'] = opt.get('topic_type', 'ldamallet')
    topic_update = opt.get('topic_update', False)
    assert topic_update in [True,False] # no  backward compatibility
    assert topic_scope in ['', 'word', 'doc', 'word+doc','word+avg']
    recover_topic_peaks = opt['unflat_topics'] =opt.get('unflat_topics', False)
    pretrained = opt.get('pretrained_embeddings', None)  # [GoogleNews,GoogleNews-reduced,Twitter,SemEval]
    w2v_limit = opt.get('w2v_limit', None)
    assert w2v_limit is None # discontinued
    calculate_mapping_rate = opt.get('mapping_rate', False)
    dim = opt.get('embedding_dim', 300)
    padding = opt.get('padding', False)
    simple_padding = opt.get('simple_padding', True)
    if padding:
        Warning('L_R_padding is discontinued. Using simple_padding instead.')
        simple_padding = True
    L_R_unk = opt.get('unk_sub', False)
    tasks = opt.get('tasks', '')
    assert len(tasks)>0
    num_topics = opt.get('num_topics',None)
    unk_topic = opt['unk_topic'] = opt.get('unk_topic', 'uniform')
    assert unk_topic in ['uniform','zero','min','small']
    lemmatize = opt.get('lemmatize',False)
    s1_max_len,s2_max_len,max_len = get_dataset_max_length(opt) #maximum number of tokens in sentence
    max_m = opt.get('max_m',None) # maximum number of examples
    bert_processing = 'bert' in opt.get('model', '') # special tokens for BERT
    injected_embd = opt.get('injected_embd',None)
    injection_embd = None

    # load or create cache
    cache = load_cache_or_process(opt, cache, onehot, vocab) # load max_m examples
    ID1 = cache['ID1']
    ID2 = cache['ID2']
    R1 = cache['R1']
    R2 = cache['R2']
    T1 = cache['T1']
    T2 = cache['T2']
    L1 = cache['L1']
    L2 = cache['L2']
    L = cache['L']

    # map words to embedding ids
    if numerical:
        if bert_processing:
            print('Mapping words to BERT ids...')
            bert_cased = opt['bert_cased'] = opt.get('bert_cased', False)
            bert_large = opt['bert_large'] = opt.get('bert_large', False)
            # use raw text rather than tokenized text as input due to different preprocessing steps for BERT
            processor_output = Preprocessor.map_files_to_bert_ids(R1, R2, s1_max_len+s2_max_len, calculate_mapping_rate,
                                                                  simple_padding=simple_padding, L_R_unk=L_R_unk,
                                                                  bert_cased=bert_cased,bert_large=bert_large)
        else:
            print('Mapping words to embedding ids...')
            processor_output = Preprocessor.map_files_to_ids(T1,T2, max_len, calculate_mapping_rate,
                                                         simple_padding=simple_padding, L_R_unk=L_R_unk)
        print('Finished word id mapping.')
        E1 = processor_output['E1']
        E1_mask = processor_output['E1_mask']
        E1_seg = processor_output['E1_seg']
        E2 = processor_output['E2']
        word2id_bert = processor_output['word2id']
        id2word_bert = processor_output['id2word']

        # load embeddings and assign
        if simple_padding:
            padding_tokens = 1
        else:
            padding_tokens = 0
        if L_R_unk:
            unk_tokens = 2
        else:
            unk_tokens = 1
        if not pretrained is None:
            vocab,embd = Preprocessor.load_embd_cache(pretrained,dataset,tasks[0], word2id_bert, w2v_limit, dim,padding_tokens,unk_tokens)

        mapping_rates = processor_output['mapping_rates']
        #  reduce embd id length for questions to save computational resources
        if not bert_processing and not s1_max_len==max_len:
            E1 = reduce_embd_id_len(E1, tasks, cutoff=s1_max_len)
        if not bert_processing and not s2_max_len==max_len:
            E2 = reduce_embd_id_len(E2, tasks, cutoff=s2_max_len)


    # reduce number of examples after mapping words to ids to ensure static mapping regardless of max_m
    if shuffle_train:
        # if we want a random subset of the data, we should shuffle before reduce_examples
        [ID1, ID2, R1, R2, T1, T2, E1, E1_mask, E1_seg, E2, D_T1, D_T2, L1, L2, L] = shuffle_train_examples([ID1, ID2, R1, R2, T1, T2, E1, E1_mask, E1_seg, E2,D_T1,D_T2, L1, L2, L])
    if not ID1 is None:
        ID1 = reduce_examples(ID1, max_m)
        ID2 = reduce_examples(ID2, max_m)
    R1 = reduce_examples(R1, max_m)
    R2 = reduce_examples(R2, max_m)
    T1 = reduce_examples(T1, max_m)
    T2 = reduce_examples(T2, max_m)
    if not E1_mask is None:
        # reduce examples for bert
        E1 = reduce_examples(E1, max_m) #[train,dev,test]
        E1_mask = reduce_examples(E1_mask, max_m)
        E1_seg = reduce_examples(E1_seg, max_m)
    elif not E1 is None:
        E1 = reduce_examples(E1, max_m)
        E2 = reduce_examples(E2, max_m)
    if not L1 is None:
        L1 = reduce_examples(L1, max_m)
        L2 = reduce_examples(L2, max_m)
    if 'doc' in topic_scope:
        # reduce doc topics here after shuffling
        D_T1 = reduce_examples(D_T1, max_m)
        D_T2 = reduce_examples(D_T2, max_m)
    L = reduce_examples(L, max_m)

    # load embd for injection
    if not injected_embd is None:
        injection_embd = load_embds(opt, add_unk = True)
        word2id_embd = injection_embd['word2id']

        # label injection test
        if 'labels' in opt['injected_embd']:
            print('Mapping words to label ids...')
            if 'inject' in opt['model']:
                # use BERT tokenizer output to ensure alignment between tokens and subword tokens (so that embd at pos i corresponds to topic distribution at pos i)
                W_T = [Preprocessor.map_subword_labels_to_id([[id2word_bert[s] for s in t] for t in r],labels, word2id_embd, s1_max_len + s2_max_len, opt) for r,labels in zip(E1,L)]

            if lemmatize:
                doc1 = L1
                doc2 = L2
            else:
                doc1 = T1
                doc2 = T2
            W_T1 = [Preprocessor.map_labels_to_id(r, labels, word2id_embd, s1_max_len, opt) for r,labels in zip(doc1,L)]
            W_T2 = [Preprocessor.map_labels_to_id(r, labels, word2id_embd, s2_max_len, opt) for r,labels in zip(doc2,L)]
        else:
            # normal case
            print('Mapping words to pretrained_embd ids...')
            if 'inject' in opt['model']:
                # use BERT tokenizer output to ensure alignment between tokens and subword tokens (so that embd at pos i corresponds to topic distribution at pos i)
                W_T = [Preprocessor.map_subword_topics_to_id([[id2word_bert[s] for s in t] for t in r], word2id_embd, s1_max_len+s2_max_len,opt) for r in E1]
                if calculate_mapping_rate:
                    injection_embd['mapping_rates'] = compute_mapping_rates(W_T)

            if lemmatize:
                doc1 = L1
                doc2 = L2
            else:
                doc1 = T1
                doc2 = T2
            W_T1 = [Preprocessor.map_topics_to_id(r,word2id_embd,s1_max_len,opt) for r in doc1]
            W_T2 = [Preprocessor.map_topics_to_id(r,word2id_embd,s2_max_len,opt) for r in doc2]

    print('Done.')
    data_dict= {'ID1': ID1, 'ID2': ID2, # doc ids
            'R1': R1, 'R2': R2, # raw text
            'T1': T1, 'T2': T2,  # tokenized text
            'L1': L1, 'L2': L2, # lemmatized text
            'E1': E1, 'E2': E2, # embedding ids
            'E1_mask': E1_mask, 'E1_seg': E1_seg,  # embedding ids
            'W_T1': W_T1, 'W_T2': W_T2, # separate word topic ids ()
            'W_T': W_T,  # joined word topic ids ()
            'D_T1':D_T1,'D_T2':D_T2, # document topics
            'L': L, # labels
            # misc
            'mapping_rates': mapping_rates,  # optional
            'embd': embd,
            'id2word':id2word_bert,
            'word2id':word2id_bert,
            'word_topics':word_topics,
            'topic_keys':topic_key_table,
            'injection_embd':injection_embd}
    return data_dict
