import math
from .Init import *
from .Test import get_hits
from scipy import sparse as sp

def get_adj(e, KG):
    row = []
    col = []
    data = []
    for tri in KG:
        h = tri[0]
        r = tri[1]
        o = tri[2]
        row.extend([h, o])
        col.extend([o, h])
        data.extend([1., 1.])
    for i in range(e):
        row.extend([i])
        col.extend([i])
        data.extend([1.])
    M = sp.coo_matrix((data, (row, col)), shape=(e,e), dtype=np.float32)
    return M


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def get_sparse_adj(e, KG):
    adj = normalize_adj(get_adj(e, KG))
    indices = np.mat([adj.row, adj.col]).transpose()
    return tf.SparseTensor(indices, adj.data, adj.shape)


# add a layer
def add_diag_layer(inlayer, dimension, M, act_func, dropout=0.0, init=ones):
    inlayer = tf.nn.dropout(inlayer, 1 - dropout)
    print('adding a layer...')
    w0 = init([1, dimension])
    tosum = tf.sparse_tensor_dense_matmul(M, tf.multiply(inlayer, w0))
    if act_func is None:
        return tosum
    else:
        return act_func(tosum)


def add_full_layer(inlayer, dimension_in, dimension_out, M, act_func, dropout=0.0, init=glorot):
    inlayer = tf.nn.dropout(inlayer, 1 - dropout)
    print('adding a layer...')
    w0 = init([dimension_in, dimension_out])
    tosum = tf.sparse_tensor_dense_matmul(M, tf.matmul(inlayer, w0))
    if act_func is None:
        return tosum
    else:
        return act_func(tosum)

# se input layer
def get_se_input_layer(e, dimension):
    print('adding the se input layer...')
    ent_embeddings = tf.Variable(tf.truncated_normal([e, dimension], stddev=1.0 / math.sqrt(e)))
    return tf.nn.l2_normalize(ent_embeddings, 1)


# ae input layer
def get_ae_input_layer(attr):
    print('adding the ae input layer...')
    return tf.constant(attr)


# get loss node
def get_loss(outlayer, ILL, gamma, k):
    print('getting loss...')
    left = ILL[:, 0]
    right = ILL[:, 1]
    t = len(ILL)
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)
    A = tf.reduce_sum(tf.abs(left_x - right_x), 1)
    neg_left = tf.placeholder(tf.int32, [t * k], "neg_left")
    neg_right = tf.placeholder(tf.int32, [t * k], "neg_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [t, k])
    D = A + gamma
    L1 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))
    neg_left = tf.placeholder(tf.int32, [t * k], "neg2_left")
    neg_right = tf.placeholder(tf.int32, [t * k], "neg2_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [t, k])
    L2 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))
    return (tf.reduce_sum(L1) + tf.reduce_sum(L2)) / (2.0 * k * t)


def highway(input_layer, units, act_func):
    H = tf.layers.dense(input_layer, units, act_func)
    T = tf.layers.dense(input_layer, units, tf.nn.sigmoid, bias_initializer=tf.constant_initializer(-1.0))
    return H*T + input_layer*(1-T)


def build_HMAN(se_dimension, act_func, gamma, k, e, ILL, KG, attr, ae_dimension, rel, rel_dimension):
    tf.reset_default_graph()
    M = get_sparse_adj(e, KG)
    # attr
    ae_layer = tf.constant(attr)
    ae_hidden = tf.layers.dense(ae_layer, ae_dimension, act_func, kernel_initializer=tf.glorot_uniform_initializer())
    ae_output = highway(ae_hidden, ae_dimension, None)
    # rel
    rel_layer = tf.constant(rel)
    rel_hidden = tf.layers.dense(rel_layer, rel_dimension, act_func, kernel_initializer=tf.glorot_uniform_initializer())
    rel_output = highway(rel_hidden, rel_dimension, None)
    # se
    se_layer = get_se_input_layer(e, se_dimension)
    se_hidden = add_diag_layer(se_layer, se_dimension, M, act_func, dropout=0.0)
    se_output = add_diag_layer(se_hidden, se_dimension, M, None, dropout=0.0)
    # fusion
    output_layer = tf.concat([se_output, ae_output, rel_output], 1)
    output_layer = tf.nn.l2_normalize(output_layer, 1)
    loss = get_loss(output_layer, ILL, gamma, k)
    return output_layer, loss


def build_MAN(se_dimension, act_func, gamma, k, e, ILL, KG, attr, ae_dimension, rel, rel_dimension):
    tf.reset_default_graph()
    M = get_sparse_adj(e, KG)
    # attr
    ae_layer = tf.constant(attr)
    ae_hidden = add_full_layer(ae_layer, attr.shape[1], ae_dimension, M, act_func, dropout=0.0)
    ae_output = add_diag_layer(ae_hidden, ae_dimension, M, None, dropout=0.0)
    # rel
    rel_layer = tf.constant(rel)
    rel_hidden = add_full_layer(rel_layer, rel.shape[1], rel_dimension, M, act_func, dropout=0.0)
    rel_output = add_diag_layer(rel_hidden, rel_dimension, M, None, dropout=0.0)
    # se
    se_layer = get_se_input_layer(e, se_dimension)
    se_hidden = add_diag_layer(se_layer, se_dimension, M, act_func, dropout=0.0)
    se_output = add_diag_layer(se_hidden, se_dimension, M, None, dropout=0.0)
    # fusion
    output_layer = tf.concat([se_output, ae_output, rel_output], 1)
    output_layer = tf.nn.l2_normalize(output_layer, 1)
    loss = get_loss(output_layer, ILL, gamma, k)
    return output_layer, loss


def training(output_layer, loss, learning_rate, epochs, ILL, e, k, valid_pair=None):
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)  # optimizer can be changed
    print('initializing...')
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.run(init)
    print('running...')
    J = []
    t = len(ILL)
    ILL = np.array(ILL)
    L = np.ones((t, k)) * (ILL[:, 0].reshape((t, 1)))
    neg_left = L.reshape((t * k,))
    L = np.ones((t, k)) * (ILL[:, 1].reshape((t, 1)))
    neg2_right = L.reshape((t * k,))
    for i in range(epochs):
        if i % 10 == 0:
            neg2_left = np.random.choice(e, t * k)
            neg_right = np.random.choice(e, t * k)

        sess.run(train_step, feed_dict={"neg_left:0": neg_left,
            "neg_right:0": neg_right,
            "neg2_left:0": neg2_left,
            "neg2_right:0": neg2_right})

        if (i + 1) % 50 == 0:
            th = sess.run(loss, feed_dict={"neg_left:0": neg_left,
                "neg_right:0": neg_right,
                "neg2_left:0": neg2_left,
                "neg2_right:0": neg2_right})
            J.append(th)
            print('%d/%d' % (i + 1, epochs), 'epochs...', ">> loss:", th)

        if valid_pair and (i + 1) % 5000 == 0:
            th = sess.run(loss, feed_dict={"neg_left:0": neg_left,
                "neg_right:0": neg_right,
                "neg2_left:0": neg2_left,
                "neg2_right:0": neg2_right})
            outvec = sess.run(output_layer)
            print("Iter@%d"%(i+1))
            get_hits(outvec, valid_pair)

    outvec = sess.run(output_layer)
    sess.close()
    return outvec, J


