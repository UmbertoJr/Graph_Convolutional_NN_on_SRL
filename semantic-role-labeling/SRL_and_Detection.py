from dataProcessing import data_generator, evaluation_generator, test_generator
from my_models import GCN
import numpy as np
import tensorflow as tf
from time import time


#### hyper-parameters and parameters
# general

batch_size= 100

max_length_sentences = 144
max_pred_in_sent = 26


gen = data_generator("../../data/CoNLL2009-ST-English-train.txt", max_len_sent=max_length_sentences)
file_eval = "../../data/CoNLL2009-ST-English-development.txt"
gen_eval = evaluation_generator(file_eval, max_len_sent=max_length_sentences, max_num_of_pred = gen.max_num_of_pred)



# input shape
dimension_POS = len(gen.pos_pos)
POS_emb_dim = 20
dimension_embeddings = 100
dimension_lemma = len(gen.lemma_pos)
lemma_emb_dim = 200

## BiLSTM and GCN
hidden_rapr_Bi = 150
input_dim_feat = 2 * hidden_rapr_Bi
num_labels_for_biases = len(gen.dep_rel_pos)

# output
total_predicates = len(gen.pred_pos)
possible_arguments = 55


############ MODEL graph
tf.reset_default_graph()
G = tf.Graph()

with G.as_default():
    with tf.name_scope("inputs"):
        ## Inputs
        with tf.name_scope("Part_of_speech"):
            one_hot_POS = tf.placeholder(tf.float32, shape = (batch_size,
                                                              max_length_sentences,
                                                              dimension_POS), 
                                         name= "POS")
            ## randomized embeddings for POS
            with tf.name_scope("random_embeddings"):
                W_emb_pos = tf.get_variable("W_emb_pos", shape = [dimension_POS ,
                                                                  POS_emb_dim], dtype = tf.float32)
                POS = tf.reshape(one_hot_POS, shape = [-1, dimension_POS ])
                POS_emb_flat = POS @ W_emb_pos    
                POS_emb = tf.reshape(POS_emb_flat, shape = [batch_size , 
                                                            max_length_sentences,
                                                            POS_emb_dim])


        with tf.name_scope("glove_embeddings"):
            embeddings = tf.placeholder(tf.float32, shape=(batch_size,
                                                           max_length_sentences ,
                                                           dimension_embeddings),
                                        name="embeddings")

            
            
        with tf.name_scope("lemma"):
            lemma_idx = tf.placeholder(tf.int32, shape = (batch_size,
                                                          max_length_sentences),
                                       name= "lemmas")
            
            with tf.name_scope("random_embeddings"):
                ## randomized embeddings for lemma
                W_emb_lemma = tf.get_variable("W_emb_lemma", shape = [dimension_lemma ,
                                                                      lemma_emb_dim], dtype = tf.float32)
                lemma = tf.reshape(lemma_idx, shape = [-1])
                lemma_emb_flat = tf.nn.embedding_lookup(W_emb_lemma, lemma)
                lemma_emb = tf.reshape(lemma_emb_flat, shape = [batch_size , 
                                                                max_length_sentences,
                                                                lemma_emb_dim])

                
              
    with tf.name_scope(""):
        indices_for_predicate = tf.placeholder(dtype=tf.int64, shape= [batch_size*max_pred_in_sent,2])
            
            
        with tf.name_scope("concat_inputs"):
            # inputs to the Bi-lstm
            inputs = tf.concat([embeddings,
                                POS_emb,
                                lemma_emb ], axis = -1)

            
            

    ######## 3 layers of Bi-Lstm
    with tf.name_scope("Bi-lstm"):
        
        bi_lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers = 3 , num_units = hidden_rapr_Bi,
                                                         direction = 'bidirectional',
                                                         dropout = 0.2,
                                                         seed = 123, dtype = tf.float32, name="Bi-Lstm")

        with tf.name_scope("output_BiLSTM"):
            output, output_states = bi_lstm(inputs)
            
           

    #############     Graph Convolutional Network layer 
    with tf.name_scope("Graph_Convolutional_Network_layer"):
        with tf.name_scope("directed_edges_and_count_labels"):
            L_i = tf.placeholder(dtype=tf.int64, shape = [batch_size, max_length_sentences, 2])
            L_v = tf.placeholder(dtype=tf.float32, shape = [batch_size, max_length_sentences])
            S_i = tf.placeholder(dtype=tf.int64, shape = [batch_size, 2*max_length_sentences, 2])
            S_v = tf.placeholder(dtype=tf.float32, shape = [batch_size, 2*max_length_sentences])

        gcnn = GCN(graph=G, batch_size = batch_size,
                   max_len= max_length_sentences, input_dim=input_dim_feat,
                   num_labels = num_labels_for_biases)   #occhio con len(d_p)

        last_layer = gcnn(inputs = output,
                          Lapl_i=L_i, Lapl_v=L_v,
                          S_i=S_i , S_v= S_v)
    
    
    #### semantic role labeling on top of GCN layer
    with tf.name_scope("Semantic_Role_Labeling-system"):
        with tf.name_scope("placeholders"):
            with tf.name_scope("mask"):
                # build mask for loss and accuracy
                with tf.name_scope("sequence_length"):
                    sequence_lengths = tf.placeholder(tf.int32, shape = [batch_size])
                    with tf.name_scope("mask-creation"):
                        s_l = tf.tile(sequence_lengths, [max_pred_in_sent])  ### copy seq_len max_pred times
                        s_l_nice = tf.transpose(tf.reshape(s_l, shape=[1,max_pred_in_sent, batch_size])) ### reshape in a nice way
                        mask_for_sequences = tf.reshape(tf.sequence_mask(s_l_nice,maxlen=max_length_sentences), 
                                                        shape = [batch_size, max_pred_in_sent, max_length_sentences])

                with tf.name_scope("count_predicate"):
                    count_pred = tf.placeholder(tf.int32, shape = [batch_size])
                    with tf.name_scope("mask-creation"):
                        c_p = tf.tile(count_pred, [max_length_sentences])
                        c_p_nice = tf.transpose(tf.reshape(c_p, shape=[1, max_length_sentences, batch_size]))
                        mask_pre = tf.sequence_mask(c_p_nice, maxlen=max_pred_in_sent)
                        mask_for_predicates = tf.transpose(tf.reshape(mask_pre,
                                                                      shape=[batch_size,max_length_sentences, max_pred_in_sent]),
                                                           perm = [0,2,1])

                
            
                with tf.name_scope("sparse_matrix_for_args"):
                    sparse_inds = tf.placeholder(dtype= tf.int64, shape=[None, 3])
                    sparse_vals = tf.placeholder(shape=[None], dtype=tf.float32)
                    S_mask = tf.SparseTensor(sparse_inds, sparse_vals,
                                             dense_shape=[batch_size, max_pred_in_sent, max_length_sentences])
                
            with tf.name_scope("predicate-position-on-sentence"):
                indices_for_predicate = tf.placeholder(dtype=tf.int64, shape= [batch_size*max_pred_in_sent, 2])
        
        with tf.name_scope("inputs"):
            with tf.name_scope("position_predicate"):
                predicate_encode = tf.gather_nd(output, indices_for_predicate)
                # just take the predicate from output of gcn layer dim: [batch*max_pred_in_sent, output_dim_encoding ]
                with tf.name_scope("final_tensor_reshape_for_concat_with_data"):
                    pred_repeated = tf.reshape (tf.tile(predicate_encode, [1,max_length_sentences]),
                                                shape = [batch_size, max_pred_in_sent*max_length_sentences, input_dim_feat])
                    # repeat each predicate encoding max_len_sent times 

            with tf.name_scope("GCN_encode_reshape_for_concatenatio"):
                output_repeated = tf.reshape( tf.tile(output,[1,max_pred_in_sent, 1]),
                                             shape=[batch_size, max_pred_in_sent*max_length_sentences, input_dim_feat])
                # repeat each sentence max_pred_in_sent times

            with tf.name_scope("concatenated_input"):
                input_to_srl_system = tf.reshape(tf.concat([pred_repeated,output_repeated], axis=2) , [-1, 2*input_dim_feat])
                # final shape is [batch*max_len_sent*max_pred_in_sent, concat_dim_encode]s
        with tf.name_scope("weights_srl_system"):
            w_srl = tf.Variable(tf.truncated_normal(shape=[2*input_dim_feat, possible_arguments],
                                                    stddev=1/(2*input_dim_feat)), name="w")
            bias_srl = tf.Variable(tf.truncated_normal(shape=[possible_arguments]),name="b")


        with tf.name_scope("predictions"):
            preds_srl_flat_logit = input_to_srl_system @ w_srl + bias_srl
            preds_srl_logit_nice = tf.reshape(preds_srl_flat_logit,
                                              shape=[batch_size, max_pred_in_sent,max_length_sentences, possible_arguments])
            predictions_srl = tf.argmax(preds_srl_logit_nice, axis=3)        
        
        with tf.name_scope("real_args"):
            real_srl_args = tf.placeholder(tf.int32, shape=[batch_size, max_length_sentences*max_pred_in_sent])
            real_srl_args_nice = tf.reshape(real_srl_args,
                                            shape=[batch_size, max_pred_in_sent, max_length_sentences])

        with tf.name_scope("final_mask"):
                    final_mask = real_srl_args_nice > 0
                    mask_seq_and_pred = mask_for_sequences & mask_for_predicates

        with tf.name_scope("loss_with_mask"):
            losses_srl = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real_srl_args_nice, logits=preds_srl_logit_nice)
            with tf.name_scope("mask_only_sentence_and_possible_predicates"):
                         losses_after_mask_arg = tf.boolean_mask(losses_srl,final_mask)
            with tf.name_scope("mask_only_arguments"):
                         losses_after_mask_arg_onl = tf.sparse_tensor_to_dense(S_mask * losses_srl)
                  
            loss_srl_args = tf.reduce_mean(losses_after_mask_arg_onl)
            loss_no_mask = tf.reduce_mean(losses_srl)
            loss_srl = tf.reduce_mean(losses_after_mask_arg)
  
        with tf.name_scope("select_interesting_variables"):
            #real_srl_aft_mask = tf.sparse_tensor_to_dense(tf.cast(S_mask, dtype=tf.int32) * real_srl_args_nice)
            #predictions_srl_aft_mask = tf.sparse_tensor_to_dense(tf.cast(S_mask, dtype=tf.int64) * predictions_srl)    
            real_srl_aft_mask = tf.boolean_mask(real_srl_args_nice, final_mask)
            predictions_srl_aft_mask = tf.boolean_mask(predictions_srl, final_mask)    
            
            
    #### DETECTION SYSTEM
    with tf.name_scope("detection-system"):
        with tf.name_scope("weights"):
            w_srl_det = tf.Variable(tf.truncated_normal(shape=[2*input_dim_feat, 2],
                                                        stddev=1/(2*input_dim_feat)), name="w")
            bias_srl_det = tf.Variable(tf.truncated_normal(shape=[2]),name="b")
            
        with tf.name_scope("is_arg"):
            arg_detection = tf.cast(final_mask, tf.int64)
        
        with tf.name_scope("detection"):
            detection_srl_flat_logit = input_to_srl_system @ w_srl_det + bias_srl_det
            detection_srl_logit_nice = tf.reshape(detection_srl_flat_logit,
                                              shape=[batch_size, max_pred_in_sent,max_length_sentences, 2])

        with tf.name_scope("loss_with_mask"):
            losses_detection = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=arg_detection, logits=detection_srl_logit_nice)
            with tf.name_scope("mask_only_sentence_and_possible_predicates"):
                         losses_after_mask_detection = tf.boolean_mask(losses_detection,mask_seq_and_pred)
            
            loss_detection = tf.reduce_mean(losses_after_mask_detection)
            
            
    ### metrics evaluation 
    with tf.name_scope("metrics_evaluation"):
        accuracy_srl, accuracy_srl_op = tf.metrics.accuracy(labels= real_srl_aft_mask, predictions = predictions_srl_aft_mask) 
        accuracy_det, accuracy_det_op = tf.metrics.accuracy(labels= arg_detection, predictions = tf.argmax(detection_srl_logit_nice, axis=3) )
        
            
            
            
    ######## OPTIMIZER
    with tf.name_scope("Optimizer") as scope:
        ### Optimizer
        lr = 0.001
        optimizer = tf.train.AdamOptimizer(lr)
        final_loss = loss_srl +  loss_detection  # 100*loss_srl_args + loss_no_mask + 1000 *
        train_op = optimizer.minimize(final_loss)



    ##### save summary for tensorboard
    with tf.name_scope("summary"):
        ## Tensorboard parameters
        tf.summary.scalar("loss-semantic-role-labeling-system", loss_srl)
        tf.summary.scalar("accuracy-score-srl", accuracy_srl)
       
        merged = tf.summary.merge_all()
    
    validation_metrics_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, 
                                                scope='metrics_evaluation/')
    validation_metrics_init_op = tf.variables_initializer(var_list=validation_metrics_vars, name='validation_metrics_init')
        

        
       

            
if tf.gfile.Exists('./train'):
    tf.gfile.DeleteRecursively('./train')
if tf.gfile.Exists('./eval'):
    tf.gfile.DeleteRecursively('./eval')

    
train_writer = tf.summary.FileWriter('./train', G)
eval_writer = tf.summary.FileWriter('./eval', G)

        


#### let's train the model
        
log_file = "./tmp/model.ckpt"

with G.as_default():    
    with tf.name_scope("trainer"):
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with open("loss_scores.txt", "w") as f:
                    f.write("#### start to train ####\n")

        start = time()
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(init_op)
            sess.run(validation_metrics_init_op)
            # start to train
            for h in range(0,10000):
                
                X_emb, X_POS, X_LEMMA, Lapl_i, Lapl_v, Sp_i, Sp_v, Pred_Ind, Arg_pos, Seq_Len, Pred_count, Sp_Ind = gen(batch_size)
                feed_dict_train={one_hot_POS: X_POS.astype(np.float32),
                                 embeddings: X_emb,
                                 lemma_idx : X_LEMMA.astype(np.int32),
                                 L_i : Lapl_i,
                                 L_v : Lapl_v.astype(np.float32),
                                 S_i : Sp_i,
                                 S_v : Sp_v.astype(np.float32),
                                 indices_for_predicate: Pred_Ind,
                                 real_srl_args : Arg_pos,
                                 sequence_lengths : Seq_Len,
                                 count_pred : Pred_count,
                                 sparse_inds : Sp_Ind,  ### da fare domani
                                 sparse_vals :  np.ones(Sp_Ind.shape[0])
                                }




                for ite in range(0,batch_size//10):
                    _ = sess.run([train_op],feed_dict=feed_dict_train)
                    
                
                if h%5 == 0:
                    
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    t_, l_srl , acc_, sum_model, l_det, acc_det = sess.run([train_op, final_loss , accuracy_srl_op, merged, loss_detection, accuracy_det_op],
                                                           feed_dict=feed_dict_train,
                                                           options=run_options, run_metadata=run_metadata)


                    print("############SAVING############")
                    #train_writer.add_run_metadata(run_metadata, 'step%03d' % (h))
                    #train_writer.add_summary(sum_model, h)
                    print('Adding run metadata for', h)
                    #save_path = saver.save(sess, log_file)
                    #print("Model saved in path: %s" % save_path)
                    ## since I'm using CudNN with no saveable object this is not useful

                    print("############TRAIN############")
                    print("loss function srl : ", l_srl, " for iteration ", (h))
                    print("accuracy srl : ", acc_)  
                    print("loss function det : ", l_det, " accuracy det ", acc_det)
                    print("execution time: ", time()-start)
                    start = time()

                    #################### evaluation of the model
                    X_emb_eval, X_POS_eval, X_LEMMA_eval, Lapl_i_eval, Lapl_v_eval, Sp_i_eval, Sp_v_eval, Pred_Ind_eval, Arg_pos_eval, Seq_Len_eval, Pred_count_eval, Sp_Ind_eval = gen_eval(batch_size)
                    feed_dict_eval={one_hot_POS: X_POS_eval.astype(np.float32),
                                    embeddings: X_emb_eval,
                                    lemma_idx : X_LEMMA_eval.astype(np.int32),
                                    L_i : Lapl_i_eval,
                                    L_v : Lapl_v_eval.astype(np.float32),
                                    S_i : Sp_i_eval,
                                    S_v : Sp_v_eval.astype(np.float32),
                                    indices_for_predicate: Pred_Ind_eval,
                                    real_srl_args : Arg_pos_eval,
                                    sequence_lengths : Seq_Len_eval,
                                    count_pred : Pred_count_eval, 
                                    sparse_inds : Sp_Ind_eval,  
                                    sparse_vals :  np.ones(Sp_Ind_eval.shape[0])
                                    }
                    l_srl_eval , acc_eval, sum_model_eval, l_det_eval, acc_det_eval = sess.run([final_loss, accuracy_srl_op, merged, loss_detection, accuracy_det_op ], feed_dict=feed_dict_eval, options=run_options, run_metadata=run_metadata) 

                    
                    eval_writer.add_run_metadata(run_metadata, 'step%03d' % (h))
                    eval_writer.add_summary(sum_model_eval, h)
                    print('Adding run metadata for', h)

                    with open("loss_scores.txt", "a") as f:
                        f.write("cross-entropy srl score= "+ str(l_srl_eval)+"\n")

                    print("############evaluation#######")
                    print("loss function srl : ", l_srl_eval, " for iteration ", (h))
                    print("accuracy srl : ", acc_eval)
                    print("loss function det : ", l_det_eval, " accuracy det ", acc_det_eval)
                    
                    

                if  h > 100 and h%30==0: 
                    
                    preds_srl, mask, detection_eval = sess.run([predictions_srl, mask_seq_and_pred, detection_srl_logit_nice], feed_dict=feed_dict_eval, options=run_options, run_metadata=run_metadata) 

                        
                    np.save("../../data/semantic-role-labeling/Cud/argument_eval.npy", preds_srl)
                    np.save("../../data/semantic-role-labeling/Cud/sequence_len_eval.npy", Seq_Len_eval)
                    np.save("../../data/semantic-role-labeling/Cud/pred_count_eval.npy", Pred_count_eval)
                    np.save("../../data/semantic-role-labeling/Cud/arg_real_eval.npy", Arg_pos_eval)
                    np.save("../../data/semantic-role-labeling/Cud/arg_ind_eval.npy", Sp_Ind_eval)
                    np.save("../../data/semantic-role-labeling/Cud/mask_eval.npy", mask)
                    np.save("../../data/semantic-role-labeling/Cud/detection_"+str(it)+"_test.npy", detection_eval)
                    print("#######evaluation data saved#############")
                    #load testverbs
                    file_test = "../../data/TestData/testverbs_with_predicate.txt"
                    gen_test = test_generator(file_test, max_len_sent=max_length_sentences)
                    # load test
                    file_test2 = "../../data/TestData/test.csv"
                    gen_test2 = test_generator(file_test2, max_len_sent=max_length_sentences)


                    ##### test numero sentence 399
                    for it in range(4):
                        
                      
                        X_emb_test, X_POS_test, X_LEMMA_test, Lapl_i_test, Lapl_v_test, Sp_i_test, Sp_v_test, Pred_Ind_test, Seq_Len_test, Pred_count_test= gen_test(batch_size)
                        
                        feed_dict_test={one_hot_POS: X_POS_test.astype(np.float32),
                                        embeddings: X_emb_test,
                                        sequence_lengths:Seq_Len_test,
                                        lemma_idx : X_LEMMA_test.astype(np.int32),
                                        L_i : Lapl_i_test,
                                        L_v : Lapl_v_test.astype(np.float32),
                                        S_i : Sp_i_test,
                                        S_v : Sp_v_test.astype(np.float32),
                                        indices_for_predicate: Pred_Ind_test,
                                        sequence_lengths : Seq_Len_test,
                                        count_pred : Pred_count_test
                                      }
                        semantic_vec_test, detection = sess.run([predictions_srl, detection_srl_logit_nice],feed_dict=feed_dict_test)
                
                        np.save("../../data/semantic-role-labeling/Cud/srl_"+str(it)+"_testverbs.npy", semantic_vec_test)
                        np.save("../../data/semantic-role-labeling/Cud/detection_"+str(it)+"_test.npy", detection)
                        
                        X_emb_test2, X_POS_test2, X_LEMMA_test2, Lapl_i_test2, Lapl_v_test2, Sp_i_test2,Sp_v_test2, Pred_Ind_test2, Seq_Len_test2, Pred_count_test2 = gen_test2(batch_size)
                        
                        feed_dict_test2={one_hot_POS: X_POS_test2.astype(np.float32),
                                         embeddings: X_emb_test2,
                                         sequence_lengths:Seq_Len_test2,
                                         lemma_idx : X_LEMMA_test2.astype(np.int32),
                                         L_i : Lapl_i_test2,
                                         L_v : Lapl_v_test2.astype(np.float32),
                                         S_i : Sp_i_test2,
                                         S_v : Sp_v_test2.astype(np.float32),
                                         indices_for_predicate: Pred_Ind_test2,
                                         sequence_lengths : Seq_Len_test2,
                                         count_pred : Pred_count_test2
                                         }
                        semantic_vec_test2, detection2 = sess.run([predictions_srl, detection_srl_logit_nice],feed_dict=feed_dict_test2)
                        
                        np.save("../../data/semantic-role-labeling/Cud/srl_"+str(it)+"_test.npy", semantic_vec_test2)
                        np.save("../../data/semantic-role-labeling/Cud/detection_"+str(it)+"_test.npy", detection2)
                        
                        print("####### test1 and test2 iteration : "+str(it)+" has been predicted ########")


            save_path = saver.save(sess, "./tmp/model.ckpt")
            print("Model saved in path: %s" % save_path)


