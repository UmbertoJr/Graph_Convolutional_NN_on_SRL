from dataProcessing import data_generator, evaluation_generator, test_generator
from my_models import GCN
import numpy as np
import tensorflow as tf
from time import time


#### hyper-parameters and parameters
# general
batch_size= 400
max_length_sentences = 144


gen = data_generator("../../data/CoNLL2009-ST-English-train.txt", max_len_sent=max_length_sentences)

file_eval = "../../data/CoNLL2009-ST-English-development.txt"
gen_eval = evaluation_generator(file_eval, max_len_sent=max_length_sentences)
file_test = "../../data/TestData/testverbs.csv"
gen_test = test_generator(file_test, max_len_sent=max_length_sentences)



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

            
            
        with tf.name_scope("concat_inputs"):
            # inputs to the Bi-lstm
            inputs = tf.concat([embeddings,
                                POS_emb,
                                lemma_emb ], axis = -1)

            
            

    ######## 2 layers of Bi-Lstm
    with tf.name_scope("Bi-lstm"):
        
        bi_lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers = 3 , num_units = hidden_rapr_Bi,
                                                         direction = 'bidirectional',
                                                         dropout = 0.2,
                                                         seed = 123, dtype = tf.float32, name="Bi-Lstm")

        with tf.name_scope("output_BiLSTM"):
            output, output_states = bi_lstm(inputs)
            
            

    #### first task to fit Predicate Detection system
    with tf.name_scope("Predicate_detection_system"):
        with tf.name_scope("input_for_predicate_detection"):
            context_flat = tf.reshape(output, shape = [-1,2*hidden_rapr_Bi ])
        
        with tf.name_scope("weight_for_detection"):
            ## Last layer weigths
            W_out = tf.get_variable("W_out", shape = [2*hidden_rapr_Bi , 2],    #the weigth is just 2 since is a classification
                                    dtype = tf.float32,
                                    initializer=tf.initializers.truncated_normal(stddev= 1/(2*hidden_rapr_Bi),
                                                                                 seed=123))
            b_out = tf.get_variable("b_out", shape = [2],
                                    dtype = tf.float32, initializer=tf.zeros_initializer())


        
        with tf.name_scope("logit_predictions"):
            pred_flat = context_flat @ W_out + b_out
            pred = tf.reshape(pred_flat, shape= [batch_size,
                                                 max_length_sentences,2])
            pred_index = tf.arg_max(pred,dimension=2)
            
        with tf.name_scope("predicate_to_detect"):
            is_pred = tf.placeholder(tf.float32, shape=[batch_size,
                                                     max_length_sentences,
                                                     2], name= "labels")


        with tf.name_scope("loss_cross_entropy"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred, labels=is_pred, name="losses_vec")
            with tf.name_scope("loss-with-mask"):
                mask = tf.sequence_mask(sequence_lengths, max_length_sentences, name="mask_for_sent_len")
                losses_det = tf.boolean_mask(losses, mask)
                loss_det = tf.reduce_mean(losses_det, name="loss")

        with tf.name_scope("accuracy"):
            my_model_det = tf.cast(tf.boolean_mask(pred_index,mask, name="predicate_detction"), dtype=tf.float32)
            real_pred = tf.cast(tf.boolean_mask(tf.argmax(is_pred,2), mask, name="real_predicate"), dtype=tf.float32)
            with tf.name_scope("correct_ones"):
                correct_det = tf.cast(tf.equal(my_model_det, real_pred), dtype=tf.float32)
            accuracy_det = tf.reduce_mean(correct_det)


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
    
    ############### Predicate Disambiguation System on top GCN
    with tf.name_scope("predicate_disambiguation_system"):
        with tf.name_scope("input_for_the_system"):
            last_layer_flat = tf.reshape(last_layer,shape=[-1, 2*hidden_rapr_Bi])
            
        with tf.name_scope("weights_and_bias"):
            weight_PD = tf.Variable(tf.truncated_normal(shape=[2*hidden_rapr_Bi,total_predicates], stddev=1/total_predicates))
            bias_PD = tf.Variable(tf.truncated_normal(shape=[total_predicates], stddev=1/total_predicates))
            
        with tf.name_scope("predicate_estiamtion"):
            pred_dis_logit_flat = last_layer_flat @ weight_PD + bias_PD
            pred_dis_logit = tf.reshape(pred_dis_logit_flat, shape=[batch_size, max_length_sentences, total_predicates])
                        
        with tf.name_scope("real_disambiguated_predicate"):
            real_pred_dis = tf.placeholder(dtype=tf.int32, shape=[batch_size, max_length_sentences])
            real_pred_dis_flat = tf.reshape(real_pred_dis, shape=[-1])
        
        with tf.name_scope("loss_cross_entropy_with_mask"):
            loss_pred_dis = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real_pred_dis, logits=pred_dis_logit)
            with tf.name_scope("mask"):
                loss_pred_dis_flat = tf.reshape(loss_pred_dis, shape=[-1])
                mask_pred_index_flat = tf.reshape(tf.arg_max(is_pred, dimension=2), shape=[-1])  
                losses_pred_dis_masked = tf.boolean_mask(loss_pred_dis_flat, mask_pred_index_flat)
            loss_pred_dis_mean = tf.reduce_mean(losses_pred_dis_masked, name="loss-mask")
            loss_all = tf.reduce_mean(loss_pred_dis, name = "loss-all")


    ######## OPTIMIZER
    with tf.name_scope("Optimizer") as scope:
        ### Optimizer
        lr = 0.001
        optimizer = tf.train.AdamOptimizer(lr)
        final_loss = 10*loss_pred_dis_mean + 100*loss_det + loss_all
        train_op = optimizer.minimize(final_loss)



    ##### save summary for tensorboard
    with tf.name_scope("summary"):
        ## Tensorboard parameters
        tf.summary.tensor_summary("cross-entropy-detection-system", losses_det)
        tf.summary.scalar("loss-detection-system", loss_det)
        tf.summary.scalar("accuracy-detection-system", accuracy_det)
        #tf.summary.tensor_summary("cross-entropy-predicate-disambiguation-system", losse_pred_dis)
        tf.summary.scalar("loss-predicate-disambiguation-system", loss_pred_dis_mean)
        tf.summary.scalar("loss total ", final_loss)
        

       
        merged = tf.summary.merge_all()

        
        
        
        
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
            
            # start to train
            for h in range(0,1000):
                X_emb, sent_len, pred_Y, pred_labels, X_POS, X_LEMMA, Lapl_i, Lapl_v, Sp_i,Sp_v = gen(batch_size)
                feed_dict_train={one_hot_POS: X_POS.astype(np.float32),
                                 embeddings: X_emb,
                                 sequence_lengths:sent_len,
                                 is_pred: pred_Y, 
                                 lemma_idx : X_LEMMA.astype(np.int32),
                                 L_i : Lapl_i,
                                 L_v : Lapl_v.astype(np.float32),
                                 S_i : Sp_i,
                                 S_v : Sp_v.astype(np.float32),
                                 real_pred_dis : pred_labels
                                 
                                }

                for ite in range(0,batch_size//10):
                    _ = sess.run([train_op],feed_dict=feed_dict_train)
                
                if h%5 == 0:
                    
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _, l_of_dete , accy_detection, l_of_disambiguation , sum_model = sess.run([train_op, loss_det,
                                                                                               accuracy_det,loss_pred_dis_mean,
                                                                                               merged],
                                                                                                feed_dict=feed_dict_train,
                                                                                                options=run_options,
                                                                                                run_metadata=run_metadata)


                    print("############SAVING############")
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % (h))
                    train_writer.add_summary(sum_model, h)
                    print('Adding run metadata for', h)
                    #save_path = saver.save(sess, log_file)
                    #print("Model saved in path: %s" % save_path)
                    ## since I'm using CudNN with no saveable object this is not useful

                    print("############TRAIN############")
                    print("loss function det : ", l_of_dete, " for iteration ", (h))
                    print("accuracy : ", accy_detection)
                    print("loss function disambiguation : ", l_of_disambiguation, " for iteration ", (h))
                    print("execution time: ", time()-start)
                    start = time()

                    #################### evaluation of the model
                    X_emb_eval, sent_len_eval, pred_Y_eval, pred_labels_eval, X_POS_eval, X_LEMMA_eval, Lapl_i_eval, Lapl_v_eval, Sp_i_eval,Sp_v_eval = gen_eval(batch_size)

                    feed_dict_eval = {one_hot_POS: X_POS_eval.astype(np.float32),
                                      embeddings: X_emb_eval,
                                      sequence_lengths:sent_len_eval,
                                      is_pred: pred_Y_eval, 
                                      lemma_idx : X_LEMMA_eval.astype(np.int32),
                                      L_i : Lapl_i_eval,
                                      L_v : Lapl_v_eval.astype(np.float32),
                                      S_i : Sp_i_eval,
                                      S_v : Sp_v_eval.astype(np.float32),
                                      real_pred_dis : pred_labels_eval
                                      }

                    pred_detection_vec, predicate_disambiguation_vec, loss_detection, loss_disamiguation, summary_eval = sess.run([pred, pred_dis_logit, loss_det, loss_pred_dis_mean, merged],feed_dict=feed_dict_eval, options=run_options, run_metadata=run_metadata)

                    #### predicate detection evaluation 
                    pred_or_not = np.argmax(pred_detection_vec, axis=2)
                    real_pred_position = np.argmax(pred_Y_eval, axis=2)
                    accuracy_detection_eval = []
                    for i in range(batch_size):
                        l = sent_len[i]
                        accuracy_detection_eval.append(np.mean(real_pred_position[i][:l] == pred_or_not[i][:l]))

                    #### predicate disambiguation evaluation 
                    pred_sense = np.argmax(predicate_disambiguation_vec, axis=2)
                    accuracy_disambiguation_eval = []
                    for i in range(batch_size):
                        l = real_pred_position[i]>0.5
                        accuracy_disambiguation_eval.append(np.mean(pred_labels_eval[i][l] == pred_sense[i][l]))


                    eval_writer.add_run_metadata(run_metadata, 'step%03d' % (h))
                    eval_writer.add_summary(summary_eval, h)
                    print('Adding run metadata for', h)

                    with open("loss_scores.txt", "a") as f:
                        f.write("cross-entropy det score= "+ str(loss_detection) + "\t cross-entropy dis score= "+ str(loss_disamiguation)+"\n")

                    print("############evaluation#######")
                    print("loss function det : ", loss_detection, " for iteration ", (h))
                    print("accuracy detection : ", np.mean(accuracy_detection_eval), " for iteration ", (h))
                    print("loss function disambiguation : ", loss_disamiguation, " for iteration ", (h))
                    print("accuracy disambiguation : ", np.mean(accuracy_disambiguation_eval), " for iteration ", (h))

                if h > 29 and h%10==0:
                    np.save("../../data/predicate-disambiguation/Cud/predicate_detection_eval.npy", pred_or_not)
                    np.save("../../data/predicate-disambiguation/Cud/predicate_real_pos_eval.npy", real_pred_position)
                    np.save("../../data/predicate-disambiguation/Cud/predicate_disambiguation_eval.npy", pred_sense)
                    np.save("../../data/predicate-disambiguation/Cud/predicate_real_sense_eval.npy", pred_labels_eval)
                    np.save("../../data/predicate-disambiguation/Cud/sentence_len_eval.npy", sent_len_eval)
                    print("#######evaluation data saved#############")
                    ##### test numero sentence 399
                    X_emb_test, sent_len_test, X_POS_test, X_LEMMA_test, Lapl_i_test, Lapl_v_test, Sp_i_test,Sp_v_test = gen_test(batch_size)
                    feed_dict_test={one_hot_POS: X_POS_test.astype(np.float32),
                               embeddings: X_emb_test,
                               sequence_lengths:sent_len_test,
                               lemma_idx : X_LEMMA_test.astype(np.int32),
                               L_i : Lapl_i_test,
                               L_v : Lapl_v_test.astype(np.float32),
                               S_i : Sp_i_test,
                               S_v : Sp_v_test.astype(np.float32),
                                  }
                    pred_detection_vec_test, predicate_disambiguation_vec_test = sess.run([pred, pred_dis_logit],
                                                                             feed_dict=feed_dict_test)

                    ### save the predicate detection for the test data
                    pred_detection_vec_test = pred_detection_vec_test[:399]
                    if_is_pred = np.argmax(pred_detection_vec_test, axis=2)
                    np.save("../../data/predicate-disambiguation/Cud/predicate_detection_test.npy", if_is_pred)

                    ### save the most probable predicate sense
                    pred_test_dis = predicate_disambiguation_vec_test[:399]
                    sorted_pred = np.argsort(pred_test_dis,axis=2)
                    useful_mat = sorted_pred[:,:,:7000:-1]
                    np.save("../../data/predicate-disambiguation/Cud/predicate_disambiguation_test.npy", useful_mat)
                    print("####### test has been predicted ########")



            save_path = saver.save(sess, "./tmp/model.ckpt")
            print("Model saved in path: %s" % save_path)


