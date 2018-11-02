from dataProcessing import data_generator, evaluation_generator, test_generator
from my_models import GCN
import numpy as np
import tensorflow as tf
from time import time
from tensorflow.keras import backend as K


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
hidden_rapr_Bi = 100
input_dim_feat = 2 * hidden_rapr_Bi
num_labels_for_biases = len(gen.dep_rel_pos)

# output
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


        with tf.name_scope("sequence_length"):
            sequence_lengths = tf.placeholder(tf.int32, shape = [batch_size])
            
            with tf.name_scope("mask-creation"):
                s_l = tf.tile(sequence_lengths, [max_pred_in_sent])  ### copy seq_len max_pred times
                s_l_nice = tf.transpose(tf.reshape(s_l, shape=[1,max_pred_in_sent, batch_size])) ### reshape in a nice way
                mask_for_sequences = tf.reshape(tf.sequence_mask(s_l_nice, maxlen=max_length_sentences), 
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


            mask_on_len_sent_and_num_pred = mask_for_predicates & mask_for_sequences            
                
                                
        with tf.name_scope("predicate-position-on-sentence"):
            indices_for_predicate = tf.placeholder(dtype=tf.int64, shape= [batch_size*max_pred_in_sent, 2])

                
    with tf.name_scope("outputs"):
        
        with tf.name_scope("real_args"):
            real_srl_args_nice = tf.placeholder(tf.int32, shape=[batch_size, max_pred_in_sent, max_length_sentences])
            

                
            
            

    ######## 1 layers of Bi-Lstm  ##############################
    with tf.name_scope("Bi-lstm"):
        ######## inputs to the Bi-lstm
        with tf.name_scope("concat_inputs"): 
            inputs_1_bi = tf.concat([embeddings,POS_emb, lemma_emb,one_hot_POS ], axis = -1)  

        bi_lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers = 1 , num_units = hidden_rapr_Bi,
                                                         direction = 'bidirectional',
                                                         dropout = 0.2,
                                                         seed = 123, dtype = tf.float32, name="Bi-Lstm")

        with tf.name_scope("output_BiLSTM"):
            output_1_bi, output_states = bi_lstm(inputs_1_bi)
            
            
            

            
    #############     Graph Convolutional Network layer #############################
    with tf.name_scope("Graph_Convolutional_Network_layer"):
        with tf.name_scope("directed_edges_and_count_labels"):
            L_i = tf.placeholder(dtype=tf.int64, shape = [batch_size, max_length_sentences, 2])
            L_v = tf.placeholder(dtype=tf.float32, shape = [batch_size, max_length_sentences])
            S_i = tf.placeholder(dtype=tf.int64, shape = [batch_size, 2*max_length_sentences, 2])
            S_v = tf.placeholder(dtype=tf.float32, shape = [batch_size, 2*max_length_sentences])

        gcnn = GCN(graph=G, batch_size = batch_size,
                   max_len= max_length_sentences, input_dim=input_dim_feat,
                   num_labels = num_labels_for_biases)  

        gcn_layer = gcnn(inputs = output_1_bi,
                          Lapl_i=L_i, Lapl_v=L_v,
                          S_i=S_i , S_v= S_v)
    

        
    ########### ATTENTION LAYER   ###################
    
    with tf.name_scope("inputs_attention"):            
            dim_input_att = tf.shape(gcn_layer)[-1]
            with tf.name_scope("gathering_predicate"):
                predicate_vec = tf.gather_nd(gcn_layer, indices_for_predicate)
                # just take the predicate from output of gcn layer dim: [batch*max_pred_in_sent, output_dim_encoding ]
               
            with tf.name_scope("repeating_the_predicate_max_len_sentence_times"):
                pred_repeated_vec = tf.reshape (tf.tile(predicate_vec , [1,max_length_sentences]),
                                                shape = [batch_size, max_pred_in_sent*max_length_sentences, dim_input_att])
                    # repeat each predicate encoding max_len_sent times 

                    
            with tf.name_scope("repeating_the_sentence_max_predicate_in_a_sentnece_times"):
                words_repeated_vec = tf.reshape( tf.tile(gcn_layer ,[1,max_pred_in_sent, 1]),
                                             shape=[batch_size, max_pred_in_sent*max_length_sentences, dim_input_att])
                # repeat each sentence max_pred_in_sent times
                words_repeated_vec_nice = tf.reshape ( words_repeated_vec, [batch_size, max_pred_in_sent, 
                                                                            max_length_sentences, dim_input_att])
                
    
    with tf.name_scope("Attention-on-predicate"):  
        
        with tf.name_scope("weigths"):
            W_att_w = tf.Variable(tf.truncated_normal((dim_input_att,1)),  name="words-weigth")
            W_att_pred = tf.Variable(tf.truncated_normal((dim_input_att,1)),  name="predicates-weigth")
            B_pos_w = tf.Variable(tf.truncated_normal([max_length_sentences]), name="bias-on-position-words")
            B_pos_pred = tf.Variable(tf.truncated_normal([max_length_sentences]), name="bias-on-position-predicate")

        with tf.name_scope("Attention_vector"):
            intermediate_vec_words = tf.reshape( tf.squeeze(tf.tensordot(words_repeated_vec, W_att_w, axes=[[2],[0]]),-1), [batch_size, 
                                                                                                              max_pred_in_sent,
                                                                                                              max_length_sentences])
            vec_words_rapr = tf.tanh( intermediate_vec_words + B_pos_w ) 
            
            
            intermediate_vec_pred = tf.reshape( tf.squeeze(tf.tensordot(pred_repeated_vec, W_att_pred, axes=[[2],[0]]),-1), [batch_size, 
                                                                                                              max_pred_in_sent,
                                                                                                              max_length_sentences])
            
            vec_preds_rapr = tf.tanh( intermediate_vec_pred + B_pos_pred)

        with tf.name_scope("similarity_and_alphas"):
            similarity = vec_words_rapr * vec_preds_rapr
            #with tf.name_scope("mask_len_sentence"):  #### masking sequence len and num predicates
                #similarity = tf.boolean_mask(similarity, mask)
            with tf.name_scope("alpha_weights"):
                a = tf.exp(similarity)
                a /= tf.reduce_sum(a, axis=2, keepdims=True) + K.epsilon()
                a = tf.expand_dims(a, -1)

        with tf.name_scope("context_vec"):
            weighted_input = words_repeated_vec_nice * a 
            context = tf.reduce_sum(weighted_input, axis=2)
            context_rep = tf.reshape(tf.tile(context,[1,1,144]),[batch_size, max_pred_in_sent,
                                                                 max_length_sentences, dim_input_att])

        with tf.name_scope("concatenation"):
            out_att = tf.reshape(tf.concat([words_repeated_vec_nice, context_rep], axis=-1), [batch_size,
                                                                                              max_pred_in_sent *max_length_sentences, 
                                                                                              2*dim_input_att])



    ######## 1 layers of Bi-Lstm  ##############################
    with tf.name_scope("Bi-lstm"):

        bi_lstm_2 = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers = 1 , num_units = hidden_rapr_Bi,
                                                         direction = 'bidirectional',
                                                         dropout = 0.1,
                                                         seed = 123, dtype = tf.float32, name="Bi-Lstm")

        with tf.name_scope("output_BiLSTM"):
            output_2_bi, output2_states = bi_lstm_2(out_att)
            
            
            
    
    
    ########### Argument Detection System
        #### DETECTION SYSTEM
    with tf.name_scope("detection-system"):
        
        with tf.name_scope("inputs_det"):
            dim_input_det = tf.shape(output_2_bi)[-1]
            input_to_det_system_flat = tf.reshape(output_2_bi, [-1,dim_input_det])
            
        
        with tf.name_scope("weights"):
            w_srl_det_hidden_2 = tf.Variable(tf.truncated_normal(shape=[dim_input_det, 100],
                                                        stddev=1/(2*tf.cast(dim_input_det, tf.float32))), name="w")
            
            
            bias_srl_det_hidden_2 = tf.Variable(tf.truncated_normal(shape=[100]),name="b")

            
            w_srl_det = tf.Variable(tf.truncated_normal(shape=[100, 2],
                                                        stddev=1/(2*tf.cast(100, tf.float32))), name="w")
            bias_srl_det = tf.Variable(tf.truncated_normal(shape=[2]),name="b")

        with tf.name_scope("is_arg"):
            arg_det_no_hot = tf.cast((real_srl_args_nice > 0), tf.int64)
            arg_detection = tf.one_hot(arg_det_no_hot, 2)
            
            
            

        with tf.name_scope("detection"):
            detection_hidden = input_to_det_system_flat @ w_srl_det_hidden_2 + bias_srl_det_hidden_2
            detection_srl_flat_logit = detection_hidden @ w_srl_det + bias_srl_det
            
            detection_srl_logit_nice = tf.reshape(detection_srl_flat_logit,
                                              shape=[batch_size, max_pred_in_sent,max_length_sentences, 2])
            
            detection_srl_pred = tf.argmax(detection_srl_logit_nice, axis=3)

            

        with tf.name_scope("loss_with_mask"):
            losses_detection_weigthted = tf.nn.weighted_cross_entropy_with_logits( targets=arg_detection ,
                                                                        logits=detection_srl_logit_nice, pos_weight = 100)
            
            losses_masked = tf.boolean_mask( losses_detection_weigthted, mask_on_len_sent_and_num_pred)
            
            loss_detection = tf.reduce_mean(losses_masked)


            with tf.name_scope("final_mask"):
                detection_mask = (real_srl_args_nice > 0) | (detection_srl_pred > 0) 
            
            with tf.name_scope("variables_of_interest"):
                detection_srl_logit_masked = tf.boolean_mask(detection_srl_logit_nice, detection_mask)            
                detection_srl_masked = tf.boolean_mask(detection_srl_pred , detection_mask)
                arg_detection_masked = tf.boolean_mask(arg_det_no_hot, detection_mask)


            
           

    #### semantic role labeling on top of GCN layer
    with tf.name_scope("Semantic_Role_Labeling-system"):        
        with tf.name_scope("inputs_srk"):
            dim_input_srl = tf.shape(output_2_bi)[-1]
            input_to_srl_system_flat = tf.reshape(output_2_bi, [-1,dim_input_det])

                
        with tf.name_scope("weights_srl_system"):
            w_srl = tf.Variable(tf.truncated_normal(shape=[dim_input_srl, possible_arguments],
                                                    stddev=1/(300.)), name="w")
            
            bias_srl = tf.Variable(tf.truncated_normal(shape=[possible_arguments]),name="b")


        with tf.name_scope("predictions"):
            preds_srl_flat_logit = input_to_srl_system_flat @ w_srl + bias_srl
            preds_srl_logit_nice = tf.reshape(preds_srl_flat_logit,
                                              shape=[batch_size, max_pred_in_sent, max_length_sentences, possible_arguments])
            predictions_srl = tf.argmax(preds_srl_logit_nice, axis=3)        

        
        
        with tf.name_scope("loss_with_mask"):
            losses_srl = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real_srl_args_nice, logits=preds_srl_logit_nice)
            with tf.name_scope("mask_on_len_and_num_preds"):
                losses_after_mask_arg = tf.boolean_mask(losses_srl,mask_on_len_sent_and_num_pred)
                  
            loss_srl = tf.reduce_mean(losses_after_mask_arg)
  

        with tf.name_scope("select_interesting_variables"):
            real_srl_aft_mask = tf.boolean_mask(real_srl_args_nice, detection_mask)
            predictions_srl_aft_mask = tf.boolean_mask(predictions_srl, detection_mask)    
            
            
 
            
    ### metrics evaluation 
    with tf.name_scope("metrics_evaluation"):
        accuracy_srl, accuracy_srl_op = tf.metrics.accuracy(labels= tf.reshape(real_srl_aft_mask,[-1]),
                                                            predictions = tf.reshape(predictions_srl_aft_mask, [-1]) )
        accuracy_det, accuracy_det_op = tf.metrics.accuracy(labels= tf.reshape( arg_detection_masked, [-1]),
                                                            predictions = tf.reshape(detection_srl_masked, [-1]) )
        
            
            
            
    ######## OPTIMIZER
    with tf.name_scope("Optimizer") as scope:
        ### Optimizer
        lr = 0.001
        optimizer = tf.train.AdamOptimizer(lr)
        final_loss = loss_srl     +  loss_detection # + 100*loss_srl_args 
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
                                 real_srl_args_nice : Arg_pos,
                                 sequence_lengths : Seq_Len,
                                 count_pred : Pred_count
                                }




                for ite in range(0,batch_size//10):
                    _ = sess.run([train_op],feed_dict=feed_dict_train)
                    
                
                if h%5 == 0:
                    
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    t_, fin_lost,  l_srl , acc_, sum_model, l_det, acc_det = sess.run([train_op, final_loss, loss_srl , accuracy_srl_op, merged, loss_detection, accuracy_det_op],
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
                    print("total loss : ", fin_lost, " for iteration ", (h))
                    print("loss function srl : ", l_srl, "accuracy srl : ", acc_)
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
                                    real_srl_args_nice : Arg_pos_eval,
                                    sequence_lengths : Seq_Len_eval,
                                    count_pred : Pred_count_eval
                                    }
                    
                    fin_lost_eval, l_srl_eval , acc_eval, sum_model_eval, l_det_eval, acc_det_eval = sess.run([final_loss, loss_srl, accuracy_srl_op, merged, loss_detection, accuracy_det_op ], feed_dict=feed_dict_eval, options=run_options, run_metadata=run_metadata) 

                    
                    eval_writer.add_run_metadata(run_metadata, 'step%03d' % (h))
                    eval_writer.add_summary(sum_model_eval, h)
                    print('Adding run metadata for', h)

                    with open("loss_scores.txt", "a") as f:
                        f.write("cross-entropy srl score= "+ str(fin_lost_eval)+"\n")

                    print("############evaluation#######")
                    print("total loss : ", fin_lost_eval, " for iteration ", (h))
                    print("loss function srl : ", l_srl_eval, "accuracy srl : ", acc_eval )
                    print("loss function det : ", l_det_eval, " accuracy det ", acc_det_eval)
                    
                    

                if   h%30==0: #h > 100 and
                    
                    
                    
                    preds_srl, f_mask, detection_eval = sess.run([predictions_srl, detection_mask , detection_srl_logit_nice], feed_dict=feed_dict_eval, options=run_options, run_metadata=run_metadata) 

                        
                    np.save("../../data/semantic-role-labeling/Cud/argument_eval.npy", preds_srl)
                    np.save("../../data/semantic-role-labeling/Cud/sequence_len_eval.npy", Seq_Len_eval)
                    np.save("../../data/semantic-role-labeling/Cud/pred_count_eval.npy", Pred_count_eval)
                    np.save("../../data/semantic-role-labeling/Cud/arg_real_eval.npy", Arg_pos_eval)
                    np.save("../../data/semantic-role-labeling/Cud/arg_ind_eval.npy", Sp_Ind_eval)
                    np.save("../../data/semantic-role-labeling/Cud/f_mask_eval.npy", f_mask)
                    np.save("../../data/semantic-role-labeling/Cud/detection_eval.npy", detection_eval)
                    print("#######evaluation data saved#############")
                    #load testverbs
                    file_test = "../../data/TestData/testverbs_with_predicate.txt"
                    gen_test = test_generator(file_test, max_len_sent=max_length_sentences)
                    # load test
                    file_test2 = "../../data/TestData/test.csv"
                    gen_test2 = test_generator(file_test2, max_len_sent=max_length_sentences)


                    ##### test numero sentence 399
                    for it in range(400//batch_size + 1):
                        
                      
                        X_emb_test, X_POS_test, X_LEMMA_test, Lapl_i_test, Lapl_v_test, Sp_i_test, Sp_v_test, Pred_Ind_test, Seq_Len_test, Pred_count_test= gen_test(batch_size)
                        
                        feed_dict_test={one_hot_POS: X_POS_test.astype(np.float32),
                                        embeddings: X_emb_test,
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
                        np.save("../../data/semantic-role-labeling/Cud/detection_"+str(it)+"_testverb.npy", detection)
                        
                        
                    for it in range(2000//batch_size + 1):
                        X_emb_test2, X_POS_test2, X_LEMMA_test2, Lapl_i_test2, Lapl_v_test2, Sp_i_test2,Sp_v_test2, Pred_Ind_test2, Seq_Len_test2, Pred_count_test2 = gen_test2(batch_size)
                        
                        feed_dict_test2={one_hot_POS: X_POS_test2.astype(np.float32),
                                         embeddings: X_emb_test2,
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


