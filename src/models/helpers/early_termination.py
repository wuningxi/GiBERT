from src.logs.training_logs import find_best_experiment_opt
from src.logs.tf_event_logs import read_previous_performance_from_tfevent

# implements early termination policy for experiment

def terminate_early(current_score,current_epoch,metric,experiment_log_path,acceptance_range=15):
    '''
    Check if model is within accepted range (e.g. 15%) of best model at current epoch (continue) or not (shut down)
    :param current_opt: opt dict from current experiment containing stopping criterion, epoch, 
    :param opt_list: list of opt from finished experiments for this experiment
    :param acceptance_range: within what percentage of best model performance, e.g. 15%
    :return: True or False
    '''
    # read logs and find best model on training set so far
    best_opt = find_best_experiment_opt(experiment_log_path,'dev',metric) # 'data/VM_logs/topic_affinity_cnn_Quora_B.json'
    if best_opt is None:
        print('[First experiment in log. Continuing...]')
        return False
    else:
        best_model_id = best_opt['id']
        if current_epoch==5:
            print('[Early termination comparing to model {}.]'.format(best_model_id))
        # get previous results from TF log
        best_at_current_epoch = read_previous_performance_from_tfevent(best_model_id,current_epoch,'dev',metric)
        if best_at_current_epoch is None:
            print('[No previous model metrics for epoch {}. Continuing...]'.format(current_epoch))
            return False
        else:
            # calculate percentage
            percentage = (100-acceptance_range)/100
            accepted_threshold = percentage*best_at_current_epoch
            if current_score< accepted_threshold:
                print('[Below accepted threshold of {}. Terminating early.]'.format(accepted_threshold))
                return True
            else:
                print('[Above accepted threshold of {}. Continuing...]'.format(round(accepted_threshold,3)))
                return False

