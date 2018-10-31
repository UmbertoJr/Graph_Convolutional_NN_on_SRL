import tensorflow as tf
from time import time 


class GCN:
    def __init__(self, graph, batch_size, max_len, input_dim, num_labels):
        self.graph = graph
        self.batch_size = batch_size
        self.max_len = max_len
        self.input_dim = input_dim
        self.num_labels = num_labels
        with self.graph.as_default():
            with tf.name_scope("weigths-for-GCN"):
                with tf.name_scope("to"):
                    self.W_to = tf.cast(tf.Variable(tf.truncated_normal(shape=[self.input_dim, self.input_dim],
                                                                stddev=(1/self.input_dim))), dtype=tf.float32)
                with tf.name_scope("from"):
                    self.W_from = tf.cast(tf.Variable(tf.truncated_normal(shape=[self.input_dim, self.input_dim],
                                                                stddev=(1/self.input_dim), dtype=tf.float64)), dtype=tf.float32)
                with tf.name_scope("identity"):
                    self.W_I = tf.cast(tf.Variable(tf.truncated_normal(shape=[self.input_dim, self.input_dim],
                                                                stddev=(1/self.input_dim))), dtype=tf.float32)
            with tf.name_scope("biases_for_labels"):
                self.B_l = tf.cast(tf.Variable(tf.truncated_normal(shape=[2*self.num_labels, self.input_dim],
                                                                stddev=(1/self.input_dim))), dtype=tf.float32)
                
    
    def __call__(self, inputs, Lapl_i, Lapl_v,  S_i, S_v):
        def fn(x):
            T = tf.SparseTensor(indices=x[1], values=x[2], dense_shape=[self.max_len, self.max_len])
            T = tf.sparse_reorder(T)
            return tf.sparse_tensor_dense_matmul(T, x[0])
        
        def fn_T(x):
            T = tf.SparseTensor(indices=x[1], values=x[2], dense_shape=[self.max_len, self.max_len])
            T = tf.sparse_reorder(T)
            T_ = tf.sparse_transpose(T)
            return tf.sparse_tensor_dense_matmul(T_, x[0])
        
        def labels_c(x):
            T = tf.SparseTensor(indices=x[0], values=x[1], dense_shape=[self.max_len, 2* self.num_labels])
            T = tf.sparse_reorder(T)
            return tf.sparse_tensor_dense_matmul(T, self.B_l)

        with self.graph.as_default():
            with tf.name_scope("graph_convolutional_network"):
                with tf.name_scope("to_operations"):
                    elems_to = (inputs, Lapl_i, Lapl_v)  ### occhio Lapl_v deve avere dtype float32
                    TX = tf.map_fn(fn, elems_to, dtype=tf.float32,name="T_dot_Xt")
                    to_ = tf.tensordot(TX, self.W_to, axes=1, name="to_contribution")
                
                with tf.name_scope("from_operations"):
                    elems_from = (inputs, Lapl_i, Lapl_v)  ### occhio Lapl_v deve avere dtype float32
                    TtX = tf.map_fn(fn_T, elems_to, dtype=tf.float32,name="Tt_dot_Xt")
                    from_ = tf.tensordot(TtX, self.W_from, axes=1, name="from_contribution")
                    
                with tf.name_scope("identity"):
                    I = tf.tensordot(inputs, self.W_I, axes=1, name="identity_contribution")
                    
                with tf.name_scope("labels_contrib"):
                    elems_lab = (S_i, S_v)  ### occhio Lapl_v deve avere dtype float32
                    SB = tf.map_fn(labels_c, elems_lab, dtype=tf.float32,name="S_dot_B")
                    
                with tf.name_scope("output_layer_gcn"):
                    output = to_ + from_ + I + SB
            return output
                
                
                