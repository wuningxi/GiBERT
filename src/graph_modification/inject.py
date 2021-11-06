import tensorflow as tf
import tensorflow.contrib.graph_editor as ge

# add new tensors/ops and connect them with existing ones

# sgv = SubGraphView
def inject_nodes(graph, location, injection_fn, args={}):
    '''
    Applies injection function after defined node location in graph
    :param graph: current graph
    :param location: name of node after which to inject new nodes as str, e.g. 'add:0'
    :param injection_fn: function defining nodes to inject
    :param args: any additional arguments for injection_fn, e.g. new external tensors
    :return: graph with injected nodes
    '''
    # find injection location in graph
    tensors = tf.contrib.graph_editor.get_tensors(graph)
    injection_point = [t.name for t in tensors].index(location)
    # tensors[injection_point] is where we want to inject new nodes

    # find outgoing connections of tensors[injection_point]
    connection_nodes = [] # nodes for which tensors[injection_point] is currently an input
    for o in tensors:
        # print(o)
        for input in o.op.inputs:
            if input.name == tensors[injection_point].name:
                print('found connection: {}'.format(o.name))
                connection_nodes.append(o)

    # apply injection function to add nodes after tensors[injection_point] within the same name_scope
    new_node = injection_fn(tensors[injection_point],**args) # tensors[injection_point] becomes input for new nodes

    if len(connection_nodes)<1:
        Warning("Didn't find any outgoing connections for node {}".format(tensors[injection_point].name))

    else:
        # make last injected node input for connection nodes of tensors[injection_point]
        connection_inputs = ge.sgv([c.op for c in connection_nodes]) # get all current inputs of connection nodes (could have additional ones we don't want to remap)
        for j in range(len(connection_inputs.inputs)):
            # print(ge.sgv(connections[0].op).remap_inputs([j]))
            if connection_inputs.remap_inputs([j]).inputs[0].name == tensors[injection_point].name: # select which input to remap
                # reroute network so that new_node becomes (selected) input for outgoing connections
                ge.connect(ge.sgv(new_node.op), ge.sgv([c.op for c in connection_nodes]).remap_inputs([0]))
                print('connected')

        # ensure that new_node now is in the input of connection nodes
        for c in connection_nodes:
            assert new_node in [i for i in c.op.inputs],'{} is not an input of {}'.format(new_node.name,c.name)

    return graph


if __name__ == '__main__':

    # toy injection function
    v = tf.constant(2, name='v')
    w = tf.constant(4, name='w')
    args = {'v':v,'w':w}
    def constant_injection(input_tensor,v,w):
        output_tensor = (input_tensor ** v) + w
        return output_tensor

    # build toy graph
    with tf.name_scope('model'):
        # with tf.name_scope('bert'):
        a = tf.constant(1,name='a')
        b = tf.constant(2,name='b')
        c = a+b
        d = tf.constant(3,name='d')
        e = c*d
        f = tf.constant(1,name='f')
        g = c*f
        z = g + e

    graph = tf.get_default_graph()
    graph = inject_nodes(graph, 'model/add:0', constant_injection, args)

    init_op = tf.global_variables_initializer()
    model_dir = "/Users/nicole/code/CQA/data/tmp/"

    # copy_graph(tf.global_variables())

    with tf.Session() as sess:

        sess.run(init_op)
        print(z.eval()) # -> 3
        # print(outputs[-1].eval()) # -> 3

        # print(new_d.eval()) # -> 42

        writer = tf.summary.FileWriter(model_dir + '/external_injection', sess.graph)
        # save_path = saver.save(sess, "/Users/nicole/code/CQA/data/tmp/model.ckpt")
        writer.add_graph(sess.graph)
        writer.close()








