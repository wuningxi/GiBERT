import os
import tensorflow as tf
# http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

def get_model_dir(opt,VM_copy=False):
    try:
        if type(opt['id'])==str:
            model_folder = opt['datapath'] + 'baseline_models/SemEval2017_task3_submissions_and_scores 3/' + opt['id'] + '/'
        else:
            if VM_copy:
                model_folder = opt['datapath'] + 'VM_models/model_{}/'.format(opt['id'])
            else:
                model_folder = opt['datapath'] + 'models/model_{}/'.format(opt['id'])
    except KeyError:
        raise KeyError('"id" and "datapath" in opt dictionary necessary for saving or loading model.')
    return model_folder

def load_model(opt, saver, sess, epoch):
    # for early stopping
    model_path = get_model_dir(opt) + 'model_epoch{}.ckpt'.format(epoch)
    saver.restore(sess,model_path)


def load_model_and_graph(opt, sess, epoch, vm_path=True):
    # for model inspection
    model_path = get_model_dir(opt,VM_copy=vm_path) + 'model_epoch{}.ckpt'.format(epoch)
    print('Loading graph...')
    new_saver = tf.train.import_meta_graph(model_path+'.meta') # load graph
    print('Loading weights...')
    new_saver.restore(sess,model_path) # restore weights


def create_saver():
    return tf.train.Saver(max_to_keep=1)

def create_model_folder(opt):
    folder = get_model_dir(opt)
    if os.path.exists(folder):
        FileExistsError('{} already exists. Please delete.'.format(folder))
    else:
        os.mkdir(folder)

def save_model(opt, saver, sess, epoch):
    model_path = get_model_dir(opt) + 'model_epoch{}.ckpt'.format(epoch)
    print(model_path)
    saver.save(sess, model_path)

def delete_all_checkpoints_but_best(opt,best_epoch):
    # list all files in model dir
    model_dir = get_model_dir(opt)
    # list all checkpoints but best
    best_model = 'model_epoch{}.ckpt'.format(best_epoch)
    to_delete = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f)) and f.startswith('model_epoch') and not f.startswith(best_model)]
    if len(to_delete)>0:
        print('Deleting the following checkpoint files:')
        for f in to_delete:
            file_path = os.path.join(model_dir, f)
            print(file_path)
            # delete
            os.remove(file_path)
