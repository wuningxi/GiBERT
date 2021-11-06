import src.loaders.build_data as build_data
import os
from src.loaders.Semeval.helper import Loader
import csv
import random

files = ['s_train','s_dev','s_test']


def reformat_split(outpath, dtype, inpath):
    print('reformatting:' + inpath)
    # Quality	#1 ID	#2 ID	#1 String	#2 String
    with open(os.path.join(outpath, dtype+'_B.txt'), 'w', encoding='utf-8') as output_file:
        with open(inpath, newline='', encoding='utf-8') as f:
            csv_reader = csv.reader(f, delimiter='\t',quoting=csv.QUOTE_NONE,quotechar='|')
            print(next(csv_reader)) # skip header
            for line in csv_reader:
                print(line)
                if len(line)==7:
                    subset, origin, year, id, labelB, doc1, doc2 = line
                elif len(line)==9:
                    subset, origin, year, id, labelB, doc1, doc2,_,_ = line
                id1 = '{}-{}-{}'.format(year,id,1)
                id2 = '{}-{}-{}'.format(year,id,2)
                output_file.write(id1 + '\t' + id2 + '\t' + doc1 + '\t' + doc2 + '\t' + labelB + '\n')


def build(opt):
    dpath = os.path.join(opt['datapath'], opt['dataset'])
    embpath = os.path.join(opt['datapath'], 'embeddings')
    logpath = os.path.join(opt['datapath'], 'logs')
    modelpath = os.path.join(opt['datapath'], 'models')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)
        build_data.make_dir(embpath)
        build_data.make_dir(logpath)
        build_data.make_dir(modelpath)

        # Download the data.
        # Use data from Kaggle
        # raise NotImplemented('Quora data not implemented yet')

        # '/Users/nicole/code/CQA/data/Quora/quora_duplicate_questions.tsv'
        # fnames = ['quora_duplicate_questions.tsv']

        # urls = ['https://zhiguowang.github.io' + fnames[0]]

        dpext = os.path.join(dpath, 'stsbenchmark')
        # build_data.make_dir(dpext)

        # for fname, url in zip(fnames,urls):
        #     build_data.download(url, dpext, fname)
        #     build_data.untar(dpext, fname) # should be able to handle zip

        reformat_split(dpath, files[0], os.path.join(dpext, 'sts-train.csv'))
        reformat_split(dpath, files[1], os.path.join(dpext, 'sts-dev.csv'))
        reformat_split(dpath, files[2], os.path.join(dpext, 'sts-test.csv'))

        # reformat(dpath, files[1], os.path.join(dpext, 'test.csv'))

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)