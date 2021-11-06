from src.models.save_load import get_model_dir
import csv
import numpy as np
import tensorflow as tf
from src.models.tf_helpers import maybe_print
import pandas as pd
from src.loaders.load_data import load_data

def activations_to_labels(activations):
    return tf.argmax(activations, 1)

def get_confidence_scores(Z3,normalised=False):
    if normalised:
        # normalise logits to get probablities??
        Z3 = tf.nn.softmax(Z3)
    conf_score = tf.gather(Z3,1,axis=1,name='conf_score') # is equal to:  Z3[:,1]
    return conf_score

def compute_evaluation_metrics(Z3, Y, map=False, sparse=True, name='evaluation'):
    """
    Calculate accuracy of predictions based on labels
    :param Z3: activations with shape (m, 2) (not normalised by softmax layer)
    :param Y: labels with shape (m, )
    :return: accuracy of predictions
    """
    with tf.name_scope(name):
        predicted_label = tf.argmax(Z3, 1, name='predict')  # which column is the one with the highest activation value?
        if sparse:
            actual_label = Y
        else:
            actual_label = tf.argmax(Y, 1)
        correct_prediction = tf.equal(predicted_label, actual_label)  # does this column equal the gold label?

        maybe_print([predicted_label,actual_label],['Predicted label','Actual label'],True)

        # Calculate evaluation metrics
        TP = tf.count_nonzero(predicted_label * actual_label, name='TP')
        # TN = tf.count_nonzero((Z3 - 1) * (Y - 1))
        FP = tf.count_nonzero(predicted_label * (actual_label - 1), name='FP')
        FN = tf.count_nonzero((predicted_label - 1) * actual_label, name='FN')

        precision = tf.divide(TP,(TP + FP),'precision')
        recall = tf.divide(TP, (TP + FN), 'recall')
        f1 = tf.divide(2 * precision * recall, (precision + recall),'f1') # macro F1
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"),name='accuracy')  # average over all examples

        tf.summary.scalar('Accuracy',accuracy)
        tf.summary.scalar('F1',f1)
        tf.summary.scalar('Precision',precision)
        tf.summary.scalar('Recall',recall)
        metrics = [accuracy,precision,recall,f1]

        # ToDo: add MAP
        if map:
            # if sparse:
            #     tf.reshape(actual_label)
            m_ap = compute_map(actual_label, Z3)
            tf.summary.scalar('MAP', m_ap)
            metrics.append(m_ap)

        conf_scores = get_confidence_scores(Z3)

    return metrics, predicted_label, conf_scores


def compute_map(labels,predictions):
    """
    Computes average precision for document batch
    :param labels: shape=(batch_size, num_labels)
    :param predictions: shape=(batch_size, num_classes)
    :return: 
    """
    try:
        labels = tf.reshape(labels, [tf.shape(labels)[0], 1])
        # assert labels.get_shape()[1] == 1
        # assert predictions.get_shape()[1] == 2
        # assert labels.get_shape()[0] == predictions.get_shape()[0]
    except Exception:
        labels = np.reshape(labels, [labels.shape[0], 1])
        # assert labels.shape[1] == 1
        # assert predictions.shape[1] == 2
        # assert labels.shape[0] == predictions.shape[0]

    # labels = np.array([[0], [0], [1], [1], [1], [0]])
    # y_true = labels.astype(np.int64)
    y_true = tf.identity(labels)

    # predictions = np.array([[0.1, 0.2],
    #           [0.8, 0.05],
    #           [0.3, 0.4],
    #           [0.6, 0.25],
    #           [0.1, 0.2],
    #           [0.9, 0.0]])
    # y_pred = predictions.astype(np.float32)
    y_pred = tf.identity(predictions)

    # _, m_ap = tf.metrics.average_precision_at_k(y_true, y_pred, 1)
    _, m_ap = tf.metrics.average_precision_at_k(y_true, y_pred, 5)
    return m_ap

def save_eval_metrics(metrics, opt, data_split='test',dict_key='score'):
    # dev_metrics = [acc, prec, rec, f_1, ma_p]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'MAP']
    if dict_key not in opt:
        opt[dict_key] = {}
    for i,eval_score in enumerate(metrics):
        metric = metric_names[i]
        if metric not in opt[dict_key]:
            opt[dict_key][metric] = {}
        if eval_score is None:
            opt[dict_key][metric][data_split] = eval_score
        else:
            opt[dict_key][metric][data_split] = round(float(eval_score), 4) # prevent problem when writing log file
    return opt

def output_predictions(query_ids,doc_ids,Z3,Y,subset,opt):
    '''
    Writes an output files with system predictions to be evaluated by official Semeval scorer.
    :param query_ids: list of question ids
    :param doc_ids: list of document ids
    :param Z3: numpy array with ranking scores (m,)
    :param Y: numpy array with True / False (m,)
    :param opt: 
    :param subset: string indicating which subset of the data ('train','dev','test)
    :return: 
    '''
    if 'PAWS' in subset:
        subset = subset.replace('PAWS','p')
    outfile = get_model_dir(opt)+'subtask'+''.join(opt['tasks'])+'.'+subset+'.pred'
    with open(outfile, 'w') as f:
        file_writer = csv.writer(f,delimiter='\t')
        # print(Y)
        label = [str(e==1).lower() for e in Y]
        for i in range(len(query_ids)):
            file_writer.writerow([query_ids[i],doc_ids[i],0,Z3[i],label[i]])

def read_predictions(opt,subset='dev',VM_path=True):
    '''
    Reads prediction file from model directory, extracts pair id, prediction score and predicted label.
    :param opt: option log
    :param subset: ['train','dev','test']
    :param VM_path: was prediction file transferred from VM?
    :return: pandas dataframe
    '''
    if type(opt['id'])==str:
        if opt['dataset']=='Semeval':
            outfile = get_model_dir(opt,VM_copy=VM_path)+'subtask_'+''.join(opt['tasks'])+'_'+subset+'.txt'
        else:
            outfile = get_model_dir(opt, VM_copy=VM_path) + 'subtask' + ''.join(opt['tasks']) + '.' + subset + '.pred'
    else:
        outfile = get_model_dir(opt,VM_copy=VM_path)+'subtask'+''.join(opt['tasks'])+'.'+subset+'.pred'
    print(outfile)
    predictions = []
    with open(outfile, 'r') as f:
        file_reader = csv.reader(f,delimiter='\t')
        for id1,id2,_,score,pred_label in file_reader:
            pairid = id1+'-'+id2
            if pred_label == 'true':
                pred_label=1
            elif pred_label == 'false':
                pred_label=0
            else:
                raise ValueError("Output labels should be 'true' or 'false', but are {}.".format(pred_label))
            predictions.append([pairid,score,pred_label])
    cols = ['pair_id','score','pred_label']
    prediction_df = pd.DataFrame.from_records(predictions,columns=cols)
    return prediction_df

def read_original_data(opt, subset='dev'):
    '''
    Reads original labelled dev file from data directory, extracts get pair_id, gold_label and sentences.
    :param opt: option log
    :param subset: ['train','dev','test']
    :return: pandas dataframe
    '''
    # adjust filenames in case of increased training data
    if 'train_large' in opt['subsets']:
        print('adjusting names')
        if subset=='dev':
            subset='test2016'
        elif subset=='test':
            subset='test2017'
    # adjust loading options:
    opt['subsets'] = [subset] # only specific subset
    opt['load_ids'] = True # with labels
#     print(opt)
    data_dict = load_data(opt,numerical=False)
    ID1 = data_dict['ID1'][0] # unlist, as we are only dealing with one subset
    ID2 = data_dict['ID2'][0]
    R1 = data_dict['R1'][0]
    R2 = data_dict['R2'][0]
    L = data_dict['L'][0]
    # extract get pair_id, gold_label, sentences
    labeled_data = []
    for i in range(len(L)):
        pair_id = ID1[i]+'-'+ID2[i]
        gold_label = L[i]
        s1 = R1[i]
        s2 = R2[i]
        labeled_data.append([pair_id,gold_label,s1,s2])
    # turn into pandas dataframe
    cols = ['pair_id','gold_label','s1','s2']
    label_df = pd.DataFrame.from_records(labeled_data,columns=cols)
    return label_df

if __name__ is '__main__':

    # opt = {'dataset': 'Semeval', 'datapath': 'data/',
    #        'tasks': ['B'],
    #        'subsets': ['test2017'],
    #        'load_ids': True, 'max_length': 100, 'max_m': 100,
    #        'id':1}
    #
    #  data = load_data(opt,cache=False)
    #  (ID1,ID2,X1_train, X2_train, Y_train), _ = data['word_ids']

    tf.reset_default_graph()
    # shape=(batch_size, num_labels)
    # order of activations and labels not important

    labels = np.array([[0],[0], [1], [0], [0]])
    label_idx = tf.expand_dims(tf.where(tf.not_equal(labels, 0))[:,0],0)
    print(label_idx.get_shape())
    # print(label_idx.eval())

    # shape=(batch_size, num_classes)
    activations= np.array([
        [0.0, 0.8],
        [0.5, 0.2],
        [0.6, 0.4],
        [0.2, 0.6],
        [0.4, 0.5]])
    print(activations.shape)
    conf_scores = tf.expand_dims(get_confidence_scores(activations),0)
    # conf_scores = tf.expand_dims(activations[:, 1],0)
    print(conf_scores)
    print(conf_scores.get_shape())
    # print(conf_scores.eval())

    metrics = tf.metrics.average_precision_at_k(label_idx, conf_scores, 5)
    # map = compute_map(tf.reshape(predicted_labels,[5,10]),tf.reshape(correct_labels,[5,10]))

    with tf.Session() as sess:

    # Z3 = create_artificial_activations(50)
    # Y = create_artificial_labels(50)

        sess.run(tf.local_variables_initializer())
        # why local??

        m= sess.run(metrics)
        print(m)
    # output_predictions(ID1,ID2,Z3,Y,'dev',opt)
