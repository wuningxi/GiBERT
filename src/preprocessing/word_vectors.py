from gensim.models import KeyedVectors,Word2Vec
import numpy as np
import time
import os
import pickle

def load_vectors(filename, word2id, limit, return_model=False, dim=300,padding_tokens=0,unk_tokens=0):
    print('Loading pretrained embeddings from file...')

    # initialise random emb
    divident = 1.0
    n = len(word2id)
    embd = np.empty((n, dim), dtype=np.float32)
    embd[:,:] = np.random.normal(size=(n,dim)) / divident
    vocab = list(word2id)

    # possibly used in pretrained embeddings
    URL_tokens = ['<url>','URLTOK']
    IMG_tokens = ['<pic>','IMG']
    # used in our preprocessing
    URL_token = '<URL>'
    IMG_token = '<IMG>'

    assert str(dim) in filename, "dimension {} is not in filename: {}".format(dim,filename)
    # assign pretrained embeddings if available
    if filename.endswith('.gz'):
        w2v = KeyedVectors.load_word2vec_format(filename, binary=True, limit=limit)
        # get embedding matrix
        try:
            pretrained_embd = w2v.syn0 # changed to prevent deprecation warning
        except AttributeError:
            pretrained_embd = w2v.wv.vectors
        pretrained_vocab = w2v.index2word
        # todo: only keep embds that are in word2id
        raise NotImplementedError()
        print('Loaded Word2Vec from {}.'.format(filename))
    elif filename.endswith('.bin'):
        w2v = Word2Vec.load(filename) # gensim version '3.0.0' needed!
        # w2v = Word2Vec.load_word2vec_format(filename,binary=True)
        # get embedding matrix
        try:
            pretrained_embd = w2v.syn0 # changed to prevent deprecation warning
        except AttributeError:
            pretrained_embd = w2v.wv.vectors
        pretrained_vocab = w2v.index2word
        # todo: only keep embds that are in word2id
        raise NotImplementedError()
        print('Loaded Word2Vec from {}.'.format(filename))
    elif filename.endswith('.npy'):
        # read embedding matrix (numpy array)
        pretrained_embd = np.load(filename)
        # read vocabulary index (pickle)
        vocab_filename = '/'.join(filename.split('/')[:-1]) + '/vocabulary.pickle'
        with (open(vocab_filename, "rb")) as openfile:
            while True:
                try:
                    pretrained_vocab=pickle.load(openfile)
                except EOFError:
                    break
        # embedding has UNK embd at end
        print('pretrained embd: {}'.format(pretrained_embd.shape))
        print('pretrained vocab: {}'.format(len(pretrained_vocab)))
        assert(pretrained_embd.shape[0]==len(pretrained_vocab)+1)
        # check special tokens and add info if possible
        if not URL_token in pretrained_vocab.keys():
            for t in URL_tokens:
                if t in pretrained_vocab.keys():
                    pretrained_vocab[URL_token]=pretrained_vocab[t]
                    print('Successfully substituted {} with {} in embd file.'.format(t,URL_token))
                    break
        if not IMG_token in pretrained_vocab.keys():
            for t in IMG_tokens:
                if t in pretrained_vocab.keys():
                    pretrained_vocab[IMG_token]=pretrained_vocab[t]
                    print('Successfully substituted {} with {} in embd file.'.format(t,IMG_token))
                    break
        for word,(index,freq) in pretrained_vocab.items():
            if word in word2id:
                # assign pretrained embeddings if available
                word_id = word2id[word]
                embd[word_id] = pretrained_embd[index]
        print('Loaded Numpy array from {}.'.format(filename))
    else:
        # plain text file with format
        # word 0.02 0.045 0.053
        inconsistent_length = []
        IMG_tokens.append('imageurl')
        with open(filename,'r',encoding='utf-8') as file:
            url_found = False
            img_found = False
            url_vec = None
            img_vec = None
            for i,line in enumerate(file.readlines()):
                if limit is None or i < limit:
                    row = line.strip().split(' ')
                    # ignore lines with inconsistent length (Twitter embedding)
                    if len(row) == dim+1:
                        # only keep if in dataset vocabulary
                        if row[0] in word2id:
                            # assign pretrained embeddings if available
                            word_id = word2id[row[0]]
                            embd[word_id] = row[1:]
                            if row[0]==URL_token:
                                url_found = True
                            elif row[0]==IMG_token:
                                img_found = True
                        # check for special tokens
                        elif row[0] in URL_tokens:
                            url_vec = row[1:]
                        elif row[0] in IMG_tokens:
                            img_vec = row[1:]
                    else:
                        inconsistent_length.append(i)
                    # vector = np.array([float(e) for e in row[1:]])
                    # if embd is None:
                    #     embd = vector
                    # else:
                    #     np.vstack((embd,vector))
        # check special tokens and add pretrained vector if possible
        if (URL_token in word2id.keys()) and (not url_found) and (not url_vec is None):
            word_id = word2id[URL_token]
            embd[word_id]=url_vec
            print('Successfully substituted url token with {} in embd file.'.format(URL_token))
        if (IMG_token in word2id.keys()) and (not img_found) and (not img_vec is None):
            word_id = word2id[IMG_token]
            embd[word_id]=img_vec
            print('Successfully substituted img token with {} in embd file.'.format(IMG_token))
        embd = np.array(embd).astype(np.float32)
        print('Loaded GloVe from {}.'.format(filename))
        if len(inconsistent_length)>0:
            print('Omitted {} rows due to inconsistent length:'.format(len(inconsistent_length)))
            print(inconsistent_length)

    # set padding vectors to zero
    pad_id = word2id['<PAD>']
    zeros = np.zeros((1, dim))
    embd[pad_id] =zeros

    assert embd.shape == (n,dim)

    if return_model:
        return vocab, embd, w2v
    else:
        return vocab, embd

def write_reduced_embedding(infile='/Users/nicole/data/embeddings/word2vec/GoogleNews-vectors-negative300.bin.gz',
                            outfile='/Users/nicole/data/embeddings/word2vec/GoogleNews-vectors-negative300-lower.bin.gz',
                            limit=None):
    '''
    Loads pretrained word2vec file and reduces its size by lowercasing words and omitting ngrams, 
    writes a new file for further usage.
    '''
    print('Reducing w2v embedding...')
    vocab, embd = load_vectors(infile, limit=limit, append=True, return_model=False)
    new_vocab = []
    new_embd = []
    old_vocab = set(vocab)
    for i,w in enumerate(vocab):
        if not '_' in w: # no ngrams
            if w.islower():
                # directly use word embeddings if lower case
                new_vocab.append(w)
                new_embd.append(embd[i])
            else:
                # ignore upper case word embedding if lower case available, otherwise use upper case word embedding for lower case word
                if w.lower() not in old_vocab:
                    w = w.lower()
                    new_vocab.append(w)
                    new_embd.append(embd[i])
            if i % 100000 == 0:
                print(i)

    print('Original embeddings: {}'.format(len(vocab)))
    print('Reduced embeddings: {}'.format(len(new_vocab)))

    ## Try to save with Gensim
    # w2v.syn0 = np.array(new_embd)
    # w2v.index2word = new_vocab
    # vocab = {}
    # for i,w in enumerate(new_vocab):
    #     vocab[w] = new_embd[i]
    # w2v.vocab = vocab
    #
    # w2v.save_word2vec_format('/Users/nicole/data/embeddings/word2vec/test', binary=True)

    with open(outfile, 'w') as f:
        for i in range(len(new_embd)):
            row = new_vocab[i] + ' ' + ' '.join([str(e) for e in new_embd[i].tolist()]) + '\n'
            f.writelines(row)
    print('Finished writing lower case unigram embeddings.')


def write_vocabulary(vocab, opt, embd_name):
    folder = opt['datapath']
    if not os.path.exists(folder):
        os.mkdir(folder)
    vocab_file = os.path.join(folder,'models',embd_name)
    if not os.path.exists(vocab_file):
        vocab = ['word_vector'] + vocab
        with open(vocab_file, 'w', encoding='utf-8') as outfile:
            for w in vocab:
                outfile.writelines(w+'\n')
        print('Wrote vocab file: {}.'.format(vocab_file))
    else:
        print('Found existing vocab file: {}'.format(vocab_file))

if __name__ == '__main__':
    # write_reduced_embedding(limit=None)
    # infile = '/Users/nicole/data/embeddings/word2vec/GoogleNews-vectors-negative300.bin.gz'
    # infile = '/Users/nicole/data/embeddings/word2vec/GoogleNews-vectors-negative300-lower.txt'
    infile = '/Users/nicole/data/embeddings/reduced_embeddings/en_embeddings_200M_200d/embedding_matrix.npy'
    # infile = 'data/embeddings/glove/glove.840B.300d.txt'
    start = time.time()
    vocab, test_embd = load_vectors(infile, limit=1000, append=False, dim=300)
    end = time.time()
    loading_time = end-start
    print(loading_time)
    # write_vocabulary(test_vocab,'twitter_100d.tsv')

