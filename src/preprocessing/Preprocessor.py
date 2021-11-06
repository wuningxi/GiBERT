# adapted from https://github.com/whiskeyromeo/CommunityQuestionAnswering

import nltk
import re
from tensorflow.contrib import learn
import numpy as np
from src.preprocessing.word_vectors import load_vectors,write_reduced_embedding
import os
import warnings
import collections
import gzip
import pickle
from pathlib import Path
from src.models.helpers.new_bert import convert_sentence_pairs_to_features,create_tokenizer,get_bert_version

def get_homedir():
    '''
    Returns user homedir across different platforms
    :return:
    '''
    return str(Path.home())

def reduce_embd_id_len(E1,tasks, cutoff=100):
    '''
    Reduces numpy array dimensions for word embedding ids to save computational resources, e.g. from (2930, 200) to (2930, 100)
    :param E1: document represented as numpy array with word ids of shape (m,sent_len)
    :param cutoff: sentence length after cutoff
    :return: shortened E1
    '''
    if len(tasks) > 1:
        raise NotImplementedError('Not implemented minimum length with multiple tasks yet.')
    # cut length of questions to 100 tokens, leave answers as is
    # only select questions
    E1_short = []
    for sub in E1:
        # reduce length (drop last 100 elements in array)
        d = np.delete(sub, np.s_[cutoff:], 1)
        E1_short.append(d)
    assert E1_short[-1].shape == (E1[-1].shape[0], 100)
    return E1_short

# same as convert_to_one_hot().T
def get_onehot_encoding(labels):
    classes = 2
    test = labels.reshape(labels.size, )
    onehotL = np.zeros((test.size, classes))
    onehotL[np.arange(test.size), test] = 1
    onehotL[np.arange(test.size), test] = 1
    return onehotL

def for_each_question(questions, function):
    for question in questions:
        function(questions[question])
        for relatedQuestion in questions[question]['related']:
            function(questions[question]['related'][relatedQuestion])

class Preprocessor:
    vocab_processor = None

    @staticmethod
    def basic_pipeline(sentences):
        # process text
        print("Preprocessor: replace urls and images")
        sentences = Preprocessor.replaceImagesURLs(sentences)
        print("Preprocessor: to lower case")
        sentences = Preprocessor.toLowerCase(sentences)
        print("Preprocessor: split sentence into words")
        sentences = Preprocessor.tokenize_tweet(sentences)
        print("Preprocessor: remove quotes")
        sentences = Preprocessor.removeQuotes(sentences)
        return sentences

    @staticmethod
    def minimal_pipeline(sentences):
        # for BERT: don't do any other processing
        sentences = Preprocessor.tokenize_tweet(sentences)
        return sentences

    @staticmethod
    def create_replacement_dict(vocab):
        '''
        Creates replacement dictionary for ngrams in vocabulary from pretrained word embeddings
        :param vocab: list of vocabulary items from pretrained embeddings
        :return: dictionary with replacements, {} for empty vocabulary
        '''
        replacement_dict = {}
        if not vocab is None:
            for token in vocab:
                match = re.match('[^_\W]+_[^_\W]+([^_\W]+)?',token)
                if match and token.count('_')<=2:
                    to_replace = token.replace('_', ' ')
                    replacement_dict[to_replace] = token
        print('Found {} n-grams in pretrained embeddings'.format(len(replacement_dict.keys())))
        return replacement_dict

    @staticmethod
    def replaceNgrams(sentences, replacement_dict):
        # ToDo: how to speed up?
        def multiple_replace(text, adict):
            # don't match punctuation (look for spaces rather than word boundaries \b)
            rx = re.compile(r'\b%s\b' % r'(?=\s)|(?<=\s)'.join(map(re.escape, adict)))
            # print(rx)
            def one_xlat(match):
                return adict[match.group(0)]
            return rx.sub(one_xlat, text)
        out = []
        for s in sentences:
            replaced = multiple_replace(s, replacement_dict)
            out.append(replaced)
        return out

    @staticmethod
    def replaceImagesURLs(sentences):
        out = []
        URL_token = '<URL>'
        IMG_token = '<IMG>'

        for s in sentences:
            s = re.sub(r'(http://)?www.*?(\s|$)', URL_token+'\\2', s) # URL containing www
            s = re.sub(r'http://.*?(\s|$)', URL_token+'\\1', s) # URL starting with http
            s = re.sub(r'\w+?@.+?\\.com.*',URL_token,s) #email
            s = re.sub(r'\[img.*?\]',IMG_token,s) # image
            s = re.sub(r'< ?img.*?>', IMG_token, s)
            out.append(s)
        return out

    @staticmethod
    def removeQuotes(sentences):
        '''
        Remove punctuation from list of strings
        :param sentences: list with tokenised sentences
        :return: list
        '''
        out = []
        for s in sentences:
            out.append([w for w in s if not re.match(r"['`\"]+",w)])
        return out

    @staticmethod
    def addBigrams(question):
        question['question_bigram_list'] = list(nltk.bigrams(question['question_words']))
        question['question_bigram_list_nostopwords'] = list(nltk.bigrams(question['question_words_nostopwords']))

    @staticmethod
    def addTrigrams(question):
        question['question_trigram_list'] = list(nltk.trigrams(question['question_words']))
        question['question_trigram_list_nostopwords'] = list(nltk.trigrams(question['question_words_nostopwords']))

    @staticmethod
    def addPartOfSpeech(question):
        question['question_words_pos'] = nltk.pos_tag(question['question_words'])
        question['question_words_pos_nostopwords'] = nltk.pos_tag(question['question_words_nostopwords'])

    @staticmethod
    def stopwordsList():
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords.append('...')
        stopwords.append('___')
        stopwords.append('<url>')
        stopwords.append('<img>')
        stopwords.append('<URL>')
        stopwords.append('<IMG>')
        stopwords.append("can't")
        stopwords.append("i've")
        stopwords.append("i'll")
        stopwords.append("i'm")
        stopwords.append("that's")
        stopwords.append("n't")
        stopwords.append('rrb')
        stopwords.append('lrb')
        return stopwords

    @staticmethod
    def removeStopwords(question):
        stopwords = Preprocessor.stopwordsList()
        return [i for i in question if i not in stopwords]

    @staticmethod
    def removeShortLongWords(sentence):
        return [w for w in sentence if len(w)>2 and len(w)<200]

    @staticmethod
    def tokenize_simple(iterator):
        return [sentence.split(' ') for sentence in iterator]
    @staticmethod
    def tokenize_nltk(iterator):
        return [nltk.word_tokenize(sentence) for sentence in iterator]
    @staticmethod
    def tokenize_tweet(iterator,strip=True):
        # tknzr = nltk.tokenize.TweetTokenizer(strip_handles=True, reduce_len=True)
        tknzr = nltk.tokenize.TweetTokenizer(strip_handles=True, reduce_len=True)
        result = [tknzr.tokenize(sentence) for sentence in iterator]
        if strip:
            result = [[w.replace(" ", "") for w in s] for s in result]
        return result
    @staticmethod
    def tokenize_tf(sentences):
        # from https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/contrib/learn/python/learn/preprocessing/text.py
        TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE)
        return [TOKENIZER_RE.findall(s) for s in sentences]
    @staticmethod
    def substitute_stopword(tokenized_sentences,unk_token='UNK'):
        stopwords = Preprocessor.stopwordsList()
        output = []
        for s in tokenized_sentences:
            output.append([unk_token if w in stopwords else w for w in s])
        return output

    @staticmethod
    def removeNonEnglish(question):
        #ToDo
        raise NotImplementedError()

    @staticmethod
    def toLowerCase(sentences):
        out = []
        special_tokens = ['UNK','<IMG>','<URL>']
        for s in Preprocessor.tokenize_tweet(sentences):
            sent =[]
            # split sentences in tokens and lowercase except for special tokens
            for w in s:
                if w in special_tokens:
                    sent.append(w)
                else:
                    sent.append(w.lower())
            out.append(' '.join(sent))
        return out

    @staticmethod
    def max_document_length(sentences,tokenizer):
        sentences = tokenizer(sentences)
        return max([len(x) for x in sentences]) # tokenised length of sentence!

    @staticmethod
    def pad_sentences(sentences, max_length,pad_token='<PAD>',tokenized=False):
        '''
        Manually pad sentences with pad_token (to avoid the same representation for <unk> and <pad>)
        :param sentences: 
        :param tokenizer: 
        :param max_length: 
        :param pad_token: 
        :return: 
        '''
        if tokenized:
            tokenized = sentences
            return [(s + [pad_token] * (max_length - len(s))) for s in tokenized]
        else:
            tokenized = Preprocessor.tokenize_tweet(sentences)
            return [' '.join(s + [pad_token] * (max_length - len(s))) for s in tokenized]

    @staticmethod
    def replaceUNK(sentences,old='<UNK_L>', new='<UNK_R>'):
        '''
        Substitute id for old UNK with id for new UNK
        :param sentences: list of sentences with word ids
        :param old: str
        :param new: str
        :return: 
        '''
        # find id of old
        id_old = Preprocessor.word2id[old]
        # find id new
        id_new = Preprocessor.word2id[new]
        # substitute
        sentences[sentences == id_old] = id_new
        return sentences

    @staticmethod
    def fitWordMapping(sentences,max_length=None,special_tokens=[]):
        word_counter = collections.Counter()
        for split in sentences:
            for sentence in split:
                word_counter.update(sentence)
        vocab = [w for w,c in word_counter.most_common()] # assign id according to frequency
        vocabulary = special_tokens+vocab
        Preprocessor.word2id = dict(zip(vocabulary, range(len(vocabulary))))
        Preprocessor.id2word = {v: k for k, v in Preprocessor.word2id.items()}
        print('Fitted vocabulary preprocessor. Longest sentence: {} tokens.'.format(max_length))

    @staticmethod
    def mapWordsToIds(sentences,unk_id=0):
        '''
        Gets list of sentences and transforms sentences to matrices with word ids
        :param sentences: list of sentences
        :param max_document_length: number of tokens in longest sentence
        :return: sentence matrix
        '''
        mapped_sentences = []
        for s in sentences:
            ids = [Preprocessor.word2id[word] if word in Preprocessor.word2id.keys() else unk_id for word in s]
            mapped_sentences.append(np.array(ids))
        sentence_matrix = np.array(mapped_sentences)
        assert len(sentence_matrix.shape)==2
        return sentence_matrix

    @staticmethod
    def mapIdsToWords(sentence_matrix):
        sentences = [[Preprocessor.id2word[w] for w in s] for s in sentence_matrix]
        # sentences = [s for s in Preprocessor.vocab_processor.reverse(sentence_matrix)]
        return sentences

    @staticmethod
    def reduce_sentence_len(r_tok,max_len):
        '''
        Reduce length of tokenised sentence
        :param r_tok: nested list consisting of tokenised sentences e.g. [['w1','w2'],['w3']]
        :param max_len: maximum length of sentence
        :return: nested list consisting of tokenised sentences, none longer than max_len
        '''
        return [s if len(s) <= max_len else s[:max_len] for s in r_tok]

    @staticmethod
    def map_topics_to_id(r_tok,word2id_dict,s_max_len,opt):
        r_red = Preprocessor.reduce_sentence_len(r_tok, s_max_len)
        r_pad = Preprocessor.pad_sentences(r_red, s_max_len, pad_token='UNK', tokenized=True)
        mapped_sentences = []
        if opt.get('stem',False):
            ps = nltk.stem.PorterStemmer()
        for s in r_pad:
            if opt.get('stem', False):
                # todo: topic preprocessing to prevent mismatch
                # data_tokenized = [[w.lower() for w in s] for s in data_tokenized]
                # if delete_stopwords:
                #     print('removing stopwords')
                #     data_tokenized = [Preprocessor.removeStopwords(s) for s in data_tokenized]
                # data_finished = [Preprocessor.removeShortLongWords(s) for s in data_tokenized]
                s = [ps.stem(w) for w in s]
            if opt.get('injected_embd',None)=='zeros':
                # all zeros
                ids = [word2id_dict['<TOKEN>'] for lemma in s]
            else:
                ids = [word2id_dict[lemma] if lemma in word2id_dict.keys() else 0 for lemma in s] # todo:fix 0 for UNK

            assert len(ids)==s_max_len, 'id len for {} should be {}, but is {}'.format(s,s_max_len,len(ids))
            mapped_sentences.append(np.array(ids))
        return np.array(mapped_sentences)

    @staticmethod
    def map_labels_to_id(r_tok,labels,word2id_dict,s_max_len,opt):
        r_red = Preprocessor.reduce_sentence_len(r_tok, s_max_len)
        r_pad = Preprocessor.pad_sentences(r_red, s_max_len, pad_token='UNK', tokenized=True)
        mapped_sentences = []
        if opt.get('stem',False):
            ps = nltk.stem.PorterStemmer()
        for s,l in zip(r_pad,labels):
            if opt.get('stem', False):
                s = [ps.stem(w) for w in s]
            if opt.get('injected_embd',None)=='labels':
                # all labels
                ids = [word2id_dict['<NEG>'] if l==0 else word2id_dict['<POS>'] for lemma in s]
            elif 'cls_labels' in opt.get('injected_embd',None):
                # feeding in labels for cls token
                ids = []
                for w in s:
                    if w == word2id_dict['[CLS]']:
                        if l==0:
                            w = word2id_dict['<NEG>']
                        else:
                            w = word2id_dict['<POS>']
                    elif w not in word2id_dict.keys():
                        w = word2id_dict['[UNK]']
                    else:
                        w = word2id_dict[w]
                    ids.append(w)
            elif 'word_labels' in opt.get('injected_embd',None):
                # feeding in labels for cls token
                special_tokens = [word2id_dict['[CLS]'],word2id_dict['[SEP]']]
                ids = []
                for w in s:
                    if not w in special_tokens:
                        if l==0:
                            w = word2id_dict['<NEG>']
                        else:
                            w = word2id_dict['<POS>']
                    elif w not in word2id_dict.keys():
                        w = word2id_dict['[UNK]']
                    else:
                        w = word2id_dict[w]
                    ids.append(w)
            else: # normal setting
                raise ValueError("'injected_embd' should be 'labels' for map_subword_labels_to_id()")

            assert len(ids)==s_max_len, 'id len for {} should be {}, but is {}'.format(s,s_max_len,len(ids))
            mapped_sentences.append(np.array(ids))
        return np.array(mapped_sentences)

    @staticmethod
    def map_subword_topics_to_id(r_tok,word2id_dict,s_max_len,opt):
        '''Takes tokenized sentence from BERT tokenizer and maps subword tokens to word topics'''
        mapped_sentences = []
        if opt.get('stem',False):
            ps = nltk.stem.PorterStemmer()
        for s in r_tok:
            s = Preprocessor.restore_tokens(s)  # restore subword tokens to tokens
            if opt.get('stem', False):
                # todo: topic preprocessing to prevent mismatch
                # data_tokenized = [[w.lower() for w in s] for s in data_tokenized]
                # if delete_stopwords:
                #     print('removing stopwords')
                #     data_tokenized = [Preprocessor.removeStopwords(s) for s in data_tokenized]
                # data_finished = [Preprocessor.removeShortLongWords(s) for s in data_tokenized]
                s = [ps.stem(w) for w in s]
            # todo: add CLS and SEP token
            if opt.get('injected_embd',None)=='zeros':
                # all zeros
                ids = [word2id_dict['<TOKEN>'] for lemma in s]
            else: # normal setting
                ids = [word2id_dict[lemma] if lemma in word2id_dict.keys() else 0 for lemma in s]
            assert len(ids)==s_max_len, 'id len for {} should be {}, but is {}'.format(s,s_max_len,len(ids))
            mapped_sentences.append(np.array(ids))
        return np.array(mapped_sentences)

    @staticmethod
    def map_subword_labels_to_id(r_tok,labels,word2id_dict,s_max_len,opt):
        '''Takes tokenized sentence from BERT tokenizer and maps subword tokens to labels for debugging'''
        mapped_sentences = []
        if opt.get('stem',False):
            ps = nltk.stem.PorterStemmer()
        for s,l in zip(r_tok,labels):
            s = Preprocessor.restore_tokens(s)  # restore subword tokens to tokens
            if opt.get('stem', False):
                # todo: topic preprocessing to prevent mismatch
                s = [ps.stem(w) for w in s]
            if opt.get('injected_embd',None)=='labels':
                # feeding in labels for words and special tokens
                ids = [word2id_dict['<NEG>'] if l==0 else word2id_dict['<POS>'] for lemma in s]
            elif 'cls_labels' in opt.get('injected_embd',None):
                # feeding in labels for cls token
                ids = []
                for w in s:
                    if w == '[CLS]':
                        if l==0:
                            w = word2id_dict['<NEG>']
                        else:
                            w = word2id_dict['<POS>']
                    elif w not in word2id_dict.keys():
                        w = word2id_dict['[UNK]']
                    else:
                        w = word2id_dict[w]
                    ids.append(w)
            elif 'word_labels' in opt.get('injected_embd',None):
                # feeding in labels for cls token
                special_tokens = ['[CLS]','[SEP]','[PAD]']
                ids = []
                for w in s:
                    if not w in special_tokens:
                        if l==0:
                            w = word2id_dict['<NEG>']
                        else:
                            w = word2id_dict['<POS>']
                    elif w not in word2id_dict.keys():
                        w = word2id_dict['[UNK]']
                    else:
                        w = word2id_dict[w]
                    ids.append(w)
            else: # normal setting
                raise ValueError("'injected_embd' should be 'labels' for map_subword_labels_to_id()")
                ids = [word2id_dict[lemma] if lemma in word2id_dict.keys() else 0 for lemma in s]
            assert len(ids)==s_max_len, 'id len for {} should be {}, but is {}'.format(s,s_max_len,len(ids))
            mapped_sentences.append(np.array(ids))
        return np.array(mapped_sentences)

    @staticmethod
    def restore_tokens(s):
        '''
        Combines subword tokens and ensures alignment by copying them,
        e.g. [CLS] am ##ro ##zi accused his brother --> [CLS] amrozi amrozi amrozi accused his brother
        '''
        i = 0
        while i < len(s):
            if s[i].startswith('##'):  # find start of separated token
                replacement = s[i - 1] + s[i][2:]
                for j in range(1,len(s[i:])):
                    if s[i + j].startswith('##'):  # find end
                        replacement += (s[i + j][2:])
                    else:
                        # replace subword tokens with token in s
                        for pos in range(-1,j):
                            s[i + pos] = replacement
                        i = i+j
                        break
            else:
                i += 1
        return s

    @staticmethod
    def lookup_word_dist(r_tok, id2topic_dist, num_topics, s_max_len,nontopic='zero',side=None):
        '''
        
        :param r_tok: 
        :param id2topic_dist: 
        :param num_topics: 
        :param s_max_len: 
        :param left: 
        :return: 
        '''
        # r_tok = Preprocessor.tokenize_tweet(r)
        if side=='right':
            factor = -1
        elif side in ['left',None]:
            factor = 1
        if nontopic =='uniform':
            substitute_vector = [factor*1/num_topics] * num_topics # flat uniform distribution for unseen words
        elif nontopic =='zero':
            substitute_vector = [0.0] * num_topics  # all topics set to zero for unseen words
        elif nontopic =='small':
            substitute_vector = [factor*1/num_topics/10] * num_topics  # flat small uniform distribution for unseen words
        elif nontopic=='min':
            min_value =  100
            for t in id2topic_dist.values():
                current_min = min(t)
                if current_min<min_value:
                    min_value=current_min
            substitute_vector = [factor*current_min] * num_topics  # flat distribution for unseen words

        mapped_sentences = []
        # reduce sentence length if len(s)>max_len
        r_red = Preprocessor.reduce_sentence_len(r_tok, s_max_len)
        r_pad = Preprocessor.pad_sentences(r_red, s_max_len, pad_token='UNK', tokenized=True)
        for s in r_pad:
            w_t = np.array([id2topic_dist.get(lemma, substitute_vector) for lemma in s]) # for one sentence (sent_len,num_topics)
            mapped_sentences.append(w_t)
        return np.array(mapped_sentences) #(m, sent_len, num_topics)

    @staticmethod
    def get_embd_file(pretrained_embedding,dim):
        if pretrained_embedding == 'GoogleNews-reduced':
            filename = 'data/embeddings/word2vec/GoogleNews-vectors-negative300-lower.txt'
            if not os.path.exists(filename):
                # create reduced embedding if not already existing
                write_reduced_embedding('data/embeddings/word2vec/GoogleNews-vectors-negative300.bin.gz', filename)
        elif pretrained_embedding == 'GoogleNews':
            filename = 'data/embeddings/word2vec/GoogleNews-vectors-negative300.bin.gz'
        elif pretrained_embedding == 'SemEval':
            filename = 'data/embeddings/qatarliving_qc_size200_win5_mincnt1_rpl_skip3_phrFalse_2016_02_25.word2vec/qatarliving_qc_size200_win5_mincnt1_rpl_skip3_phrFalse_2016_02_25.word2vec.bin'
        elif pretrained_embedding == 'Twitter':
            filename = 'data/embeddings/glove.twitter.27B/glove.twitter.27B.{}d.txt'.format(dim)
        elif pretrained_embedding == 'Glove_lower':
            # common crawl, lower cased
            filename = 'data/embeddings/glove/glove.42B.{}d.txt'.format(dim)
        elif pretrained_embedding == 'Glove':
            # common crawl, upper and lower case
            filename = 'data/embeddings/glove/glove.840B.{}d.txt'.format(dim)
        elif pretrained_embedding == 'Deriu':
            filename = 'data/embeddings/reduced_embeddings/en_embeddings_200M_200d/embedding_matrix.npy'
        else:
            raise ValueError("Invalid value for pretrained_embeddings. Choose one of the following: ['GoogleNews',"
                             "'GoogleNews-reduced','Twitter','Glove','Glove_lower']")
        return filename

    @staticmethod
    def load_embd_cache(pretrained_embedding,dataset,task, word2id, w2v_limit=None, dim=300, padding_tokens=0, unk_tokens=0):
        # check if exists otherwise save as preprocessed .gz
        embd_file = Preprocessor.get_embd_file(pretrained_embedding,dim)
        if dataset=='Semeval':
            embd_cache = '{}_{}{}_cache.gz'.format('.'.join(embd_file.split('.')[:-1]),dataset,task) # different embd cache for each Semeval task
        else:
            embd_cache = '{}_{}_cache.gz'.format('.'.join(embd_file.split('.')[:-1]),dataset)
        if os.path.exists(embd_cache):
            print('Loading cached embeddings from {} for {} from {}.'.format(embd_file,dataset, embd_cache))
            f = gzip.open(embd_cache, 'rb')
            embd = pickle.load(f)
            f.close()
            vocab  = list(word2id)
            assert len(vocab)==embd.shape[0]
        else:
            vocab, embd = Preprocessor.load_vocab_embd(pretrained_embedding, word2id, w2v_limit, dim, padding_tokens, unk_tokens)
            print('Saving cached embeddings from {} for {}.'.format(embd_file,dataset))
            f = gzip.open(embd_cache, 'wb')
            pickle.dump(embd, f)
            f.close()
        return vocab,embd

    @staticmethod
    def load_vocab_embd(pretrained_embedding,word2id,w2v_limit=None,dim=300,padding_tokens=0,unk_tokens=0):
        # fit with pretrained embeddings
        if pretrained_embedding is None:
            vocab = []
            embd = None
        else:
            embd_file = Preprocessor.get_embd_file(pretrained_embedding,dim)
            vocab, embd = load_vectors(embd_file, word2id, w2v_limit, False, dim,padding_tokens,unk_tokens)
        return vocab, embd

    @staticmethod
    def map_files_to_bert_ids(T1,T2, max_length, calculate_mapping_rate=False, simple_padding=False,L_R_unk=False,bert_cased=False,bert_large=False):
        '''
        Split raw text into tokens and map to embedding ids for all subsets
        :param T1: nested list with tokenized sentences in each subset e.g. [R1_train,R1_dev,R1_test]
        :param T1: nested list with tokenized sentences in each subset e.g. [R2_train,R2_dev,R2_test]
        :param max_length: number of tokens in longest sentence, int
        :param pretrained_embedding: use mapping from existing embeddings?, boolean
        :param padding_tokens: padding tokens to use, should be ['<PAD>'] or ['<PAD_L>','<PAD_R>']
        :param L_R_unk: use different UNK token in left and right sentence
        :return: {'E1':E1,'E2':E2, 'mapping_rates':mapping_rates or None}
        '''
        padding_tokens = []
        # assert (padding_tokens == ['<PAD_L>', '<PAD_R>']) or (padding_tokens == ['<PAD>'] or padding_tokens == [])
        #     # ToDo: only fit on training files!!
        # fit vocabulary
        # add special tokens to vocabulary
        special_tokens = []
        if L_R_unk:
            special_tokens.append('<UNK_L>')
            special_tokens.append('<UNK_R>')
        else:
            special_tokens.append('[UNK]')
        if simple_padding:
            padding_tokens = ['[PAD]']
            special_tokens.extend(padding_tokens)

        # pad and map
        mapping_rates = [] #todo: fix mapping rates

        # set unused to None rather than []
        E1 = []
        E1_mask = []
        E1_seg = []
        # use new_bert preprocessing code to encode sentence pairs
        for S1,S2 in zip(T1,T2): # look through subsets
            BERT_version = get_bert_version(bert_cased, bert_large)
            if bert_cased:
                lower=False
            else:
                lower=True
            tokenizer = create_tokenizer('{}/tf-hub-cache/{}/vocab.txt'.format(get_homedir(),BERT_version),
                                         do_lower_case=lower)
            Preprocessor.word2id = tokenizer.vocab  # dict(zip(vocabulary, range(len(vocabulary))))
            Preprocessor.id2word = {v: k for k, v in Preprocessor.word2id.items()}
            # S1 = [' '.join(s) for s in S1]  # don't use tokenized version
            # S2 = [' '.join(s) for s in S2]  # don't use tokenized version
            input_ids_vals, input_mask_vals, segment_ids_vals = convert_sentence_pairs_to_features(S1,S2, tokenizer,max_seq_len=max_length) # double length due to 2 sentences
            assert input_ids_vals.shape == input_mask_vals.shape == segment_ids_vals.shape
            E1.append(input_ids_vals)
            E1_mask.append(input_mask_vals)
            E1_seg.append(segment_ids_vals)

        if not calculate_mapping_rate:
            mapping_rates = None
        return {'E1':E1,'E1_mask':E1_mask,'E1_seg':E1_seg,'E2':None, 'mapping_rates':mapping_rates, 'word2id':Preprocessor.word2id,'id2word':Preprocessor.id2word}

    @staticmethod
    def map_files_to_ids(T1,T2, max_length, calculate_mapping_rate=False, simple_padding=False,L_R_unk=False):
        '''
        Split raw text into tokens and map to embedding ids for all subsets
        :param T1: nested list with tokenized sentences in each subset e.g. [R1_train,R1_dev,R1_test]
        :param T1: nested list with tokenized sentences in each subset e.g. [R2_train,R2_dev,R2_test]
        :param max_length: number of tokens in longest sentence, int
        :param pretrained_embedding: use mapping from existing embeddings?, boolean
        :param padding_tokens: padding tokens to use, should be ['<PAD>'] or ['<PAD_L>','<PAD_R>']
        :param L_R_unk: use different UNK token in left and right sentence
        :return: {'E1':E1,'E2':E2, 'mapping_rates':mapping_rates or None}
        '''
        padding_tokens = []
        # assert (padding_tokens == ['<PAD_L>', '<PAD_R>']) or (padding_tokens == ['<PAD>'] or padding_tokens == [])
        #     # ToDo: only fit on training files!!
        # fit vocabulary
        # add special tokens to vocabulary
        special_tokens = []
        if L_R_unk:
            special_tokens.append('<UNK_L>')
            special_tokens.append('<UNK_R>')
        else:
            special_tokens.append('<UNK>')
        if simple_padding:
            padding_tokens = ['<PAD>']
            special_tokens.extend(padding_tokens)
        if len(T1)==4:
            # don't fit on extra test set (e.g. PAWS)
            Preprocessor.fitWordMapping(T1[:3]+T2[:3],max_length,special_tokens)
        else:
            Preprocessor.fitWordMapping(T1+T2,max_length,special_tokens)

        # pad and map
        mapping_rates = [] #todo: fix mapping rates

        # set unused to None rather than []
        E1 = []
        E2 = []
        for subset in T1:
            output = Preprocessor.pad_and_map_subset(subset, max_length, calculate_mapping_rate,L_R_unk, padding_tokens, right_sentence=False)
            E1.append(output['subset'])
            if not output['mapping_rate'] is None:
                mapping_rates.append(output['mapping_rate'])
        for subset in T2:
             output = Preprocessor.pad_and_map_subset(subset, max_length, calculate_mapping_rate,L_R_unk, padding_tokens, right_sentence=True)
             E2.append(output['subset'])
             if not output['mapping_rate'] is None:
                 mapping_rates.append(output['mapping_rate'])
        if not calculate_mapping_rate:
            mapping_rates = None
        return {'E1':E1,'E1_mask':None,'E1_seg':None,'E2':E2, 'mapping_rates':mapping_rates, 'word2id':Preprocessor.word2id,'id2word':Preprocessor.id2word}

    @staticmethod
    def pad_and_map_subset(subset,max_length,calculate_mapping_rate, L_R_unk,padding_tokens,right_sentence):
        mapping_rate = None
        # simple_padding:
        token = padding_tokens[0]
        subset_reduced = Preprocessor.reduce_sentence_len(subset, max_length)
        subset_padded = Preprocessor.pad_sentences(subset_reduced, max_length, pad_token=token,tokenized=True)
        if calculate_mapping_rate:
            tokens = [len(s) for s in subset_padded] # ' '.split()
            expected_word_ids = sum(tokens)
        if right_sentence and L_R_unk:
            unk_id = 1
        else:
            unk_id = 0
        subset_mapped = Preprocessor.mapWordsToIds(subset_padded,unk_id)
        if calculate_mapping_rate:
            actual_word_ids = np.count_nonzero(subset_mapped)
            if calculate_mapping_rate:
                mapping_rate = actual_word_ids / expected_word_ids
        # replace UNK token in second sentence
        # if L_R_unk and right_sentence:
        #     subset_mapped = Preprocessor.replaceUNK(subset_mapped, old='<UNK_L>', new='<UNK_R>')
        return {'subset':subset_mapped,'mapping_rate':mapping_rate}


# if __name__ == '__main__':
#
#     # word topics
#     opt = {'dataset': 'Semeval', 'datapath': 'data/',
#            'tasks': ['A'],
#            'subsets': ['train_large','test2016','test2017'],
#            'model': 'basic_cnn', 'load_ids':True,
#            'num_topics':20, 'topic_type':'ldamallet'}
#     # infer_and_write_word_topics(opt)
#     processed_files = load_data(opt, cache=True, numerical=False, onehot=False)
#
#     word_topic_dict, id_word_dict = read_word_topics(opt)
#
#     lda_preprocess(samples, id2word=id_word_dict, print_steps=False) # not working:
# Traceback (most recent call last):
#   File "/Users/nicole/.virtualenvs/tensorflow/lib/python3.6/site-packages/IPython/core/interactiveshell.py", line 2910, in run_code
#     exec(code_obj, self.user_global_ns, self.user_ns)
#   File "<ipython-input-23-2ff5bc0e8804>", line 1, in <module>
#     sample_lemmas = lda_preprocess(samples, id2word=id_word_dict, print_steps=False)
#   File "<ipython-input-19-78558d80c176>", line 115, in lda_preprocess
#     corpus = [id2word.doc2bow(text) for text in texts]
#   File "<ipython-input-19-78558d80c176>", line 115, in <listcomp>
#     corpus = [id2word.doc2bow(text) for text in texts]
# AttributeError: 'dict' object has no attribute 'doc2bow'

