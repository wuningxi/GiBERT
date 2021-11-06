from src.loaders.load_data import load_data
from src.models.base_model_bert import model,test_opt
import argparse

# run aibert model

parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument('-datasets', action="store", dest="datasets", type=str, default='MSRP')
parser.add_argument('-learning_rate', action="store", dest="learning_rate", type=str, default='3e-5')
parser.add_argument('-epochs', action="store", dest="epochs", type=int, default=3)
parser.add_argument('-location', action="store", dest="location", type=str, default='embd')
parser.add_argument('-gpu', action="store", dest="gpu", type=int, default=-1)
parser.add_argument('-seed', action="store", dest="seed", type=str, default='fixed')
parser.add_argument("--debug",type="bool",nargs="?",const=True,default=False,help="Try to use small number of examples for troubleshooting")
parser.add_argument("--early_stopping",type="bool",nargs="?",const=True,default=False)
parser.add_argument('-embd_type', action="store", dest="embd_type", type=str, default='counter_fitted')
parser.add_argument("--update",type="bool",nargs="?",const=True,default=False)

FLAGS, unparsed = parser.parse_known_args()

if len(unparsed)>0:
    print(unparsed)
    parser.print_help()
    raise ValueError('Unidentified command line arguments passed: {}\n'.format(str(unparsed)))

datasets = FLAGS.datasets.split(',')
print(datasets)
for d in datasets:
    assert d in ['MSRP','Semeval_A','Semeval_B','Semeval_C','Quora']

todo = []

for d in datasets:
    stopping_criterion = None #'F1'
    patience = None
    batch_size = 32 # standard minibatch size
    if 'Semeval' in d:
        d,task = d.split('_')
        subsets = ['train_large', 'test2016', 'test2017']
        if task in ['A','C']:
            batch_size = 16 # need smaller minibatch to fit on GPU due to long sentences
    else:
        task = 'B'
        if d=='Quora':
            subsets = ['train', 'dev', 'test','p_test'] # also evaluate on PAWS
            task = 'B'
        else:
            subsets = ['train', 'dev', 'test'] # MSRP
            task = 'B'

    if FLAGS.debug:
        max_m = 10
    else:
        max_m = None

    predict_every_epoch = False

    if FLAGS.early_stopping:
        patience = 2
        stopping_criterion = 'F1'
        epochs = None
    else:
        stopping_criterion = None
        patience = None
        epochs = FLAGS.epochs

    try:
        seed = int(FLAGS.seed)
    except:
        seed = None

    opt = {'dataset': d, 'datapath': 'data/',
                         'model': 'bert_mha_inject_embd','bert_update':True,'bert_cased':False,
                         'injection_location': FLAGS.location,
                         'tasks': [task],
                         'subsets': subsets,'seed':seed,
                         'minibatch_size': batch_size, 'L2': 0,
                         'max_m': max_m, 'load_ids': True,
                        'embd_scope':'word', 'inj_embd_update':FLAGS.update,
                        'unk_embd': 'random','cls_embd': 'random', 'sep_embd': 'random',
                        'injected_embd':FLAGS.embd_type,
                        'unk_sub': False, 'padding': False, 'simple_padding': True,
                        'learning_rate': float(FLAGS.learning_rate),
                         'num_epochs': epochs,
                         'sparse_labels': True, 'max_length': 'minimum',
                       'optimizer': 'Adam', 'dropout':0.1,
                       'gpu': FLAGS.gpu,
                       'lemmatize': False,
                       'freeze_thaw_tune':False,
                       'predict_every_epoch': predict_every_epoch,'unk_embd_std':0.02,
                       'stopping_criterion':stopping_criterion, 'patience':patience
           }
    todo.append(opt)

if __name__ == '__main__':

    for i,opt in enumerate(todo):
        print('Starting experiment {} of {}'.format(i+1,len(todo)))
        test_opt(opt)
        data = load_data(opt, cache=True, write_vocab=False)
        opt = model(data, opt, logfile='aibert.json', print_dim=True)
