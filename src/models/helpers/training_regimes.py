import tensorflow as tf

def standard_training_regime(optimizer_choice, cost, learning_rate, epsilon, rho):
    # normal setting with only one learning rate and optimizer for all variables
    with tf.name_scope('train'):
        if optimizer_choice == 'Adam':
            train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        elif optimizer_choice == 'Adadelta':
            train_step = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, epsilon=epsilon, rho=rho).minimize(cost)
        else:
            raise NotImplementedError()
    return train_step

def standard_training_regime_debug_apply_gradients(optimizer_choice, cost, learning_rate, epsilon, rho):
    '''
    implements  normal setting with only one learning rate and optimizer for all variables using freeze thaw tune subfunctions
    testing showed results are similar to optimizer.minimize(cost)
    '''
    with tf.name_scope('train'):
        if optimizer_choice == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer_choice == 'Adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, epsilon=epsilon, rho=rho)
        else:
            raise NotImplementedError()
    trainable_vars = tf.trainable_variables()
    grads = tf.gradients(cost, trainable_vars)
    train_step = optimizer.apply_gradients(zip(grads, trainable_vars))
    return train_step

def layer_specific_regime(optimizer_choice, cost, learning_rate_old_layers, learning_rate_new_layers, epsilon, rho):
    '''
    Using layer-specific learning rates for BERT vs. newer layers which can be changed during training (e.g. freeze --> unfreeze)
    :param optimizer_choice: Adam or Adadelta
    :param cost: cost tensor
    :param learning_rate_old_layers: placeholder
    :param learning_rate_new_layers: placeholder
    :param epsilon:
    :param rho:
    :return: update op (combining bert optimizer and new layer optimizer)
    '''
    # based on https://stackoverflow.com/questions/34945554/how-to-set-layer-wise-learning-rate-in-tensorflow
    trainable_vars = tf.trainable_variables() # huge list of trainable model variables
    # separate existing and new variables based on name (not position as previously)
    bert_vars = []
    new_vars = []
    for t in trainable_vars:
        if t.name.startswith('bert_lookup/'):
            bert_vars.append(t)
        else:
            new_vars.append(t)
    # create optimizers with different learning rates
    with tf.name_scope('train'):
        if optimizer_choice == 'Adam':
            old_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_old_layers, name='old_optimizer')
            new_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_new_layers, name='new_optimizer')
        elif optimizer_choice == 'Adadelta':
            old_optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate_old_layers, epsilon=epsilon, rho=rho,name='old_optimizer')
            new_optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate_new_layers, epsilon=epsilon, rho=rho,name='new_optimizer')
        else:
            raise NotImplementedError()
        # only compute gradients once
        grads = tf.gradients(cost, bert_vars + new_vars)
        # separate gradients from pretrained and new layers
        bert_grads = grads[:len(bert_vars)]
        new_grads = grads[len(bert_vars):]
        # apply optimisers to respective variables and gradients
        train_step_bert = old_optimizer.apply_gradients(zip(bert_grads, bert_vars))
        train_step_new = new_optimizer.apply_gradients(zip(new_grads, new_vars))
        # combine to one operation
        train_step = tf.group(train_step_bert, train_step_new)
    return train_step

# def standard_training_regime_debug_2_optimizers(optimizer_choice, cost, learning_rate, epsilon, rho):
#     '''
#     implements normal setting with only one learning rate and optimizer for all variables using freeze thaw tune subfunctions
#     defining two training steps with same settings as standard_training_regime
#     run train_step_1 for one epoch, train_step_2 for two epochs and compare results to standard training regime
#     '''
#     with tf.name_scope('train'):
#         if optimizer_choice == 'Adam':
#             optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#         elif optimizer_choice == 'Adadelta':
#             optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, epsilon=epsilon, rho=rho)
#         else:
#             raise NotImplementedError()
#     trainable_vars = tf.trainable_variables()
#     grads = tf.gradients(cost, trainable_vars)
#     train_step_1 = optimizer.apply_gradients(zip(grads, trainable_vars))
#     train_step_2 = optimizer.apply_gradients(zip(grads, trainable_vars))
#     return train_step_1, train_step_2

# def freeze_thaw_tune_regime(optimizer_choice, cost, learning_rate, epsilon, rho,speed_up=False):
#     trainable_vars = tf.trainable_variables() # huge list of trainable model variables
#     bert_vars = []
#     new_vars = []
#     # separate existing and new variables based on name (not position as previously)
#     for t in trainable_vars:
#         if t.name.startswith('bert_lookup/'):
#             bert_vars.append(t)
#         else:
#             new_vars.append(t)
#     # create optimizers with different learning rates
#     if optimizer_choice == 'Adam':
#         with tf.name_scope('slow_optimizer'):
#             slow_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='slow_optimizer')
#         if speed_up:
#             with tf.name_scope('fast_optimizer'):
#                 fast_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate * 100, name='fast_optimizer')
#     elif optimizer_choice == 'Adadelta':
#         with tf.name_scope('slow_optimizer'):
#             slow_optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, epsilon=epsilon, rho=rho, name='slow_optimizer')
#         if speed_up:
#             with tf.name_scope('fast_optimizer'):
#                 fast_optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate * 100, epsilon=epsilon, rho=rho, name='fast_optimizer')
#     else:
#         raise NotImplementedError()
#     # only compute gradients once
#     grads = tf.gradients(cost, bert_vars + new_vars)
#     # separate gradients from pretrained and new layers
#     # bert_grads = grads[:len(bert_vars)]
#     new_grads = grads[len(bert_vars):]
#     # apply fast and slow optimizers
#     if speed_up:
#         freeze_train_step = fast_optimizer.apply_gradients(zip(new_grads, new_vars))  # apply normal learning rate only to new layers (freeze BERT)
#     else:
#         freeze_train_step = slow_optimizer.apply_gradients(zip(new_grads, new_vars))  # apply normal learning rate only to new layers (freeze BERT)
#     train_step = slow_optimizer.apply_gradients(zip(grads, bert_vars + new_vars))  # apply slow learning rate to everything (thaw BERT and finetune)
#     return freeze_train_step, train_step
