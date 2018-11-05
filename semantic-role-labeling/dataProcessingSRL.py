"""
this library is made just for SRL system
each other system has his own dataProcessing functions
"""

import numpy as np
import pickle

class read_file:
    ### This class read file and his methods return all dictionaries useful for the data manipulations
    ### and for the following class
    def __init__(self, name_file):
        self.name = name_file
        self.position_file_byte = 0
    
    def read_sentences(self, num_sent):
        ## this method return the n (num_sent) sentences in the format of list of list [[sent_1],[sent_2],...,[sent_n]]
        ## where each sent is a list of words sent = [[word_1],[word_2], ...., [word_k]]
        ### and each word has all features of CoNLL data: word = ["id", "lemma", "plemma",...]
        with open(self.name, "r") as f:
            f.seek(self.position_file_byte)
            docs = []
            sentence = []
            m = 0
            line = f.readline()
            while len(docs) < num_sent:
                if line == "":
                        f.seek(0)
                        line = f.readline()
                word = line.split("\t")
                if word[0] != "\n":
                      if  int(word[0]) > m:
                            sentence.append(word)
                            line = f.readline()
                            m += 1

                else:
                    m = 0
                    docs.append(sentence)
                    sentence = []
                    self.position_file_byte = f.tell()
                    line = f.readline()
                    
                    
            
    
        return docs
    
    def print_sentence(self, sent):
        #simple function to just print the full sentence
        print(*[sent[i][1]for i in range(len(sent))], sep=" ")
        
        
    def find_all_predicates(self, occ_greater_than):
        ## simple function that returns 3 dictionaries
        ## dic_final is a dic where key is the predicate (ex: like.01) and the value the position on the one_hot vector
        ## dic_final_to_fit is a dic ... key predicate with count greater than occ_greater_than and value the count
        ##  dic_final_not_to_fit is a dic ... key predicate with count less than occ_greater_than and value the count
        dic = {}
        with open(self.name, "r") as f:
            while True:
                line = f.readline()
                if line =="":
                    break   
                if  line.split("\t")[0]=="\n":
                    continue
                predicate = line.split()[13]
                if predicate in dic:
                    dic[predicate] +=1
                else:
                    dic[predicate] = 1

        dic_final_to_fit = {el for el in dic if dic[el]> occ_greater_than}
        dic_final_not_to_fit = {el for el in dic if dic[el] <= occ_greater_than}
        dic_final = {}
        dic_final["unk"] = 0 
        j=1
        for el in dic_final_to_fit:
            dic_final[el] = j
            j+=1
        return dic_final, dic_final_to_fit, dic_final_not_to_fit
    
    
    
    def find_all_POS(self):
        dic = {}
        with open(self.name, "r") as f:
            while True:
                line = f.readline()
                if line =="":
                    break   
                if  line.split("\t")[0]=="\n":
                    continue
                pos = line.split()[4]
                if pos in dic:
                    dic[pos] += int(bool(line.split()[13]!= "_"))
                else:
                    dic[pos] = int(bool(line.split()[13]!= "_"))
        pos_pos = {}
        j = 0
        for i in dic.keys():
            pos_pos[i] = j
            j+= 1
        return dic, pos_pos
    
    def find_all_lemmas(self):
        dic = {}
        with open(self.name, "r") as f:
            while True:
                line = f.readline()
                if line =="":
                    break   
                if  line.split("\t")[0]=="\n":
                    continue
                vec = line.split()
                lemma = vec[2]
                if lemma in dic:
                    for i in range(13, len(vec)):
                        dic[lemma] += int(bool(line.split()[i]!= "_")) 
                else:
                    for i in range(13, len(vec)):
                        dic[lemma] = int(bool(line.split()[i]!= "_"))
        lemma_pos = {}
        j = 0       ### ho fatto un cambiamento qui che 
        lemma_pos['unk'] = j
        for i in dic.keys():
            if dic[i]!=0:
                j+= 1
                lemma_pos[i] = j
            
        return dic, lemma_pos


    
    def max_length_sentence(self):
        m = 0
        j = 0
        with open(self.name, "r") as f:
            while True:
                line = f.readline()
                j += 1
                if line =="":
                    break   
                if  line.split("\t")[0]=="\n":
                    if j >m :
                        m = j
                    j = 0
                    continue
        return m
        
    def length_sentences(self):
        dic = {}
        j = 0
        with open(self.name, "r") as f:
            while True:
                line = f.readline()
                j += 1
                if line =="":
                    break   
                if  line.split("\t")[0]=="\n":
                    if j in dic:
                        dic[j] += 1
                        j = 0
                    else:
                        dic[j] = 1
                        j = 0
                    continue

        return dic
    
    
    def find_all_dependency_relations(self):
        dic = {}
        with open(self.name, "r") as f:
            while True:
                line = f.readline()
                if line =="":
                    break   
                if  line.split("\t")[0]=="\n":
                    continue
                vec = line.split()
                dep_rel = vec[10]
                if dep_rel in dic:
                    for i in range(13, len(vec)):
                        dic[dep_rel] += int(bool(line.split()[i]!= "_")) 
                else:
                    for i in range(13, len(vec)):
                        dic[dep_rel] = int(bool(line.split()[i]!= "_"))
        dep_rel_pos = {}
        dep_rel_pos["unk"] = 0
        j = 1
        for i in dic:
            dep_rel_pos[i] = j
            j+= 1
        return dic, dep_rel_pos

        
    
    def find_all_args(self):
        dic = {}
        with open(self.name, "r") as f:
            while True:
                line = f.readline()
                if line =="":
                    break   
                if  line.split("\t")[0]=="\n":
                    continue
                vec = line.split()
                for arg in range(14, len(vec) ):
                    arg = vec[arg]
                    if arg in dic:
                        dic[arg] += 1
                    else:
                        dic[arg] = 1
        arg_pos = {}
        j = 0
        arg_pos['_'] = j
        for i in dic.keys():
            if i!="_" and i != "_\n":
                j+= 1
                arg_pos[i] = j

        return dic, arg_pos
    
    def max_pred_in_sentence(self):
        m = 0
        j = 0
        with open(self.name, "r") as f:
            while True:
                line = f.readline()
                if line =="":
                    break   
                if  line.split("\t")[0]=="\n":
                    if j >m :
                        m = j
                    j = 0
                    continue
                vec = line.split()
                if vec[13]!="_":
                    j += 1
                    
        return m



    
    
class model_builder:
    # costruisce il DATASET con le parole sotto forma di vettore e definisce il modello con gli embeddings
    
    def __init__(self, pos_pos,lemma_pos, dep_rel_pos, pred_pos, arg_pos, max_length, max_num_of_pred):
        self.model = {}
        self.name_file = ""
        self.max_length = max_length
        self.max_num_of_pred = max_num_of_pred 
        self.len_pos = len(pos_pos) ## numero di possibili POS   
        self.pos_pos = pos_pos
        self.len_lemma = len(lemma_pos)
        self.lemma_pos = lemma_pos
        self.dep_rel_pos = dep_rel_pos
        self.len_dep_rel = len(self.dep_rel_pos)
        self.pred_pos = pred_pos
        self.len_pred = len(self.pred_pos)
        self.arg_pos = arg_pos
        self.len_arg = len(self.arg_pos)
        
    def load_model(self, name_file):
        ### salva gli embeddings dal file
        ## ossia crea un dizionario con key il lemma e valore il byte nel file (in maniera tale da accedere al file più rapidamente)
        self.name_file = name_file
        with open(name_file, "r") as f:
            pos = f.tell()
            while True:
                line = f.readline()
                if line == "":
                    break
                word = line.split()[0]
                self.model[word] = pos
                pos = f.tell()
                
        return self.model
    
    def __call__(self, word):
        ### tramite una chiamata ritorna l'embeddings della parola cercata
        with open(self.name_file, "r") as f:
            f.seek(self.model[word])
            return np.array(f.readline().split()[1:], dtype=np.float)
        
   
    def create_we_for(self, sent):
        #ritorna per ciascuna sentence un array dim (len_sentence + pad, dim_embeddings) 
        # [len_sentence + pad = max_lenght(valore sccelto 70)]
        #   e un intero con valore la lunghezza della sentence "n"
        n = len(sent)
        w=0
        Sent = []
        while w < len(sent) :
            lemma = sent[w][2]
            try:
                Sent.append(self(lemma))
            except:
                Sent.append(self("unk"))
            w+=1           
            
        sent_x = np.pad(np.array(Sent, dtype = np.float32),((0,self.max_length-n),(0,0)), "constant", constant_values = 0 )
        return (sent_x ,  n)
                
    
    
    def creation_for_SRL(self, sents, max_length = ""):
        ## ritorna per un batch di sentence un array dim [batch, max_length, dim_embeddings]
        # e un array dim [batch] con la lunghezza len_max di ciascuna sentence
        if max_length != "":
            self.max_length = max_length
        ## input glove embedding
        X = []
        ## len of each sequence
        seq_len = []
        ## POS and lemma one hot
        POS = []
        LEMMA = []
  
        # graph convolutional inputs
        Lapl_ind, Lapl_val = [], []
        S_ind, S_val = [], []
        
        ## SRL input and count
        PRED_IND = []
        ARG_POS = []
        
        for sent in sents:
            sent = self.change_indices(sent)
            pred_ind, pred_count = self.position_predicate_indices(sent)
            if pred_count > 0:
                for pred_num in range(pred_count):
                    x, n = self.create_we_for(sent)
                    pos, lem = self.pos_lemma_one_hot(sent)
                    arg_pos = self.position_argument_vec(pred_num, sent)
                    l_i, l_v, s_i, s_v = self.laplacian_and_S(sent)

                    X.append(x), seq_len.append(n), POS.append(pos), LEMMA.append(lem)
                    Lapl_ind.append(l_i), Lapl_val.append(l_v), S_ind.append(s_i), S_val.append(s_v)
                    ARG_POS.append(arg_pos)
                PRED_IND.extend(pred_ind)

        return np.array(X), np.array(POS), np.array(LEMMA), np.array(Lapl_ind), np.array(Lapl_val), np.array(S_ind), np.array(S_val), np.array(PRED_IND), np.array(ARG_POS), np.array(seq_len)
    
    
    def creation_for_test_srl(self, sents, max_length = ""):
        ## ritorna per un batch di sentence un array dim [batch, max_length, dim_embeddings]
        # e un array dim [batch] con la lunghezza di ciascuna sentence
        if max_length != "":
            self.max_length = max_length
            
        X = []
        seq_len = []
        POS = []
        LEMMA = []
        Lapl_ind = []
        Lapl_val = []
        S_ind = []
        S_val = []
        ## SRL input and count
        PRED_IND, PRED_count = [], []
        
        for sent in sents:
            sent = self.change_indices(sent)
            pred_ind, pred_count = self.position_predicate_indices(sent)
            if pred_count > 0:
                for pred_num in range(pred_count):
                    x, n = self.create_we_for(sent)
                    pos, lem = self.pos_lemma_one_hot(sent)
                    l_i, l_v, s_i, s_v = self.laplacian_and_S(sent)
                    X.append(x), seq_len.append(n), POS.append(pos), LEMMA.append(lem)
                    Lapl_ind.append(l_i), Lapl_val.append(l_v), S_ind.append(s_i), S_val.append(s_v)

                PRED_IND.extend(pred_ind)

        return np.array(X), np.array(POS), np.array(LEMMA), np.array(Lapl_ind), np.array(Lapl_val), np.array(S_ind), np.array(S_val), np.array(PRED_IND), np.array(seq_len)
    
    
    def change_indices(self, sent_c):
        #### this function just change the indices of inputs
        first_index = int(sent_c[0][0])
        new_sent = []
        for w in sent_c:
            new_w = w.copy()
            to = str( int(w[0]) - first_index )
            new_w[0] = to
            head = int(new_w[8]) - first_index 
            if head >= 0 and head < self.max_length:
                new_w[8] = str(head)
            else:
                new_w[8] = ""
                

            new_sent.append(new_w)
        return new_sent
    
    
    def split_the_sentence(self, sent):
        ind_pred_med, n = 0, len(sent)
        distance = n
        max_len = self.max_length
        one = self.change_indices(sent[:max_len])
        three = self.change_indices(sent[(n - max_len):])
        if n - max_len <= 50:
            return [one, three]
        else:
            two = self.change_indices(sent[50 : 50 + max_len])

            return [one, two, three]

    
    def position_predicate_indices(self, sent):
        sent_pos_pred = []
        w_pos = 0
        for w in sent:
            if w[12]=="Y":
                sent_pos_pred.append([ w_pos ])
            w_pos += 1
        n = len(sent_pos_pred)
        return sent_pos_pred, n
        
    
    def position_argument_vec(self, pred_num, sent):
        arg_in_sent = np.zeros( (self.max_length)  )
        pos_in_the_list = 14 + pred_num
        w_pos = 0
        for w in sent:
            arg = w[pos_in_the_list].split()[0]
            if arg!="_" and arg!="_\n":
                if arg in self.arg_pos:
                    arg_in_sent[w_pos] = self.arg_pos[arg]
                else:
                    print("problem with ", arg)
                    arg_in_sent[ w_pos] = 0
            w_pos += 1

        return arg_in_sent

    
    
    def pos_lemma_one_hot(self, sent):
        vec_pos = []
        sent_pos = []
        vec_lemma = []
        n = len(sent)
        for w in range(n):
            if sent[w][4]!='_':
                vec_pos.append(self.pos_pos[sent[w][4]])
                sent_pos.append(w)
            if sent[w][2] in self.lemma_pos:
                vec_lemma.append(self.lemma_pos[sent[w][2]])
            else:
                vec_lemma.append(0)
                
        one_hot_pos = np.zeros((self.max_length, self.len_pos ))
        one_hot_pos[ sent_pos , vec_pos] = 1
        one_hot_lemma = np.pad(np.array(vec_lemma), (0, self.max_length - len(sent)), "constant")
        return one_hot_pos, one_hot_lemma
    

   

    def laplacian_and_S(self, sent):
        # this fun create the Laplacian matrix and the sparse matrix S
        ## both are saved as a list of indexes and value for sparse rapresentation
        ## first part of the algorithm build a dictionary of dictionary {i:{j: number_of_relation_seen}}
        ## with i and j row and column index
        l = self.len_dep_rel
        L = {}
        S = {}
        for w in sent:
            if w[8].isdigit():
                i = int(w[8])
                
            else:
                continue
            j = int(w[0])
            
            if w[10] in self.dep_rel_pos:
                lab = int(self.dep_rel_pos[w[10]])
            else:
                lab = 0
            if i in L:
                L[i].append(j)
            else:
                L[i] = [j]     

            if i in S:
                if lab in S[i]:
                    S[i][lab] += 1

                else : 
                    S[i][lab] = 1

            else:
                S[i] = {lab : 1}

            if j in S:
                if l + lab in S[j]:
                    S[j][l + lab] += 1

                else : 
                    S[j][l + lab] = 1

            else:
                S[j] = {l + lab : 1}

        S_ind = []
        S_val = []
        for k in S:
            for i in S[k]:
                S_ind.append([k,i])
                S_val.append(S[k][i])
        S_ind.extend([[0,0] for _ in range(2*self.max_length- len(S_ind))])
        S_val.extend([0 for _ in range(2*self.max_length- len(S_val))])

        Lapl_ind = []
        Lapl_val = []
        for k in L:
            for i in L[k]:
                Lapl_ind.append([k,i])
                Lapl_val.append(1)
        Lapl_ind.extend([[0,0] for _ in range(self.max_length - len(Lapl_ind))])  
        Lapl_val.extend([0 for _ in range(self.max_length - len(Lapl_val))])

        return Lapl_ind, Lapl_val, S_ind , S_val

            
       
    
    

        
class data_generator:
    def __init__(self, file, max_len_sent):
        self.file = file
        self.reader = read_file(self.file)
        self.pos_pos = self.reader.find_all_POS()[1]
        pickle.dump(self.pos_pos, open("../../data/semantic-role-labeling/Cud/part_of_speech.pickle","wb"), protocol=pickle.HIGHEST_PROTOCOL)
        self.lemma_pos = self.reader.find_all_lemmas()[1]
        pickle.dump(self.lemma_pos, open("../../data/semantic-role-labeling/Cud/lemma.pickle","wb"), protocol=pickle.HIGHEST_PROTOCOL)
        self.dep_rel_pos = self.reader.find_all_dependency_relations()[1]
        pickle.dump(self.dep_rel_pos, open("../../data/semantic-role-labeling/Cud/dependecy_relation.pickle","wb"), protocol=pickle.HIGHEST_PROTOCOL)
        self.max_length = max_len_sent
        self.pred_pos = self.reader.find_all_predicates(0)[0]
        pickle.dump(self.pred_pos, open("../../data/semantic-role-labeling/Cud/predicate.pickle","wb"), protocol=pickle.HIGHEST_PROTOCOL)
        self.arg_pos = self.reader.find_all_args()[1]
        pickle.dump(self.arg_pos, open("../../data/semantic-role-labeling/Cud/arguments.pickle","wb"), protocol=pickle.HIGHEST_PROTOCOL)
        self.max_num_of_pred = self.reader.max_pred_in_sentence()
        self.model = model = model_builder(self.pos_pos, 
                                           self.lemma_pos,
                                           self.dep_rel_pos,
                                           self.pred_pos, 
                                           self.arg_pos,
                                           max_length = self.max_length,
                                           max_num_of_pred = self.max_num_of_pred
                                          )
        self.list_of_words = self.model.load_model("../../data/glove.6B.100d.txt")
        self.batch = 0
    
    def __call__(self, batch_size):
        self.batch = batch_size
        sents = self.reader.read_sentences(batch_size)
        X_emb, X_POS, X_LEMMA, Lapl_i, Lapl_v, Sp_i, Sp_v, Pred_Ind, Arg_Pos , Seq_Len = self.model.creation_for_SRL(sents)
        ind = np.random.choice(np.arange(X_emb.shape[0]), size=batch_size, replace=False)
        #ind = np.arange(batch_size)
        return X_emb[ind], X_POS[ind], X_LEMMA[ind], Lapl_i[ind], Lapl_v[ind], Sp_i[ind], Sp_v[ind], Pred_Ind[ind], Arg_Pos[ind], Seq_Len[ind]
    

    
    
    
    
    
    
class evaluation_generator:
    def __init__(self, file, max_len_sent, max_num_of_pred):
        self.file = file
        self.reader = read_file(self.file)
        self.pos_pos = pickle.load(open("../../data/semantic-role-labeling/Cud/part_of_speech.pickle","rb"))
        self.lemma_pos = pickle.load(open("../../data/semantic-role-labeling/Cud/lemma.pickle","rb"))
        self.dep_rel_pos = pickle.load(open("../../data/semantic-role-labeling/Cud/dependecy_relation.pickle","rb"))
        self.max_length = max_len_sent
        self.pred_pos = pickle.load(open("../../data/semantic-role-labeling/Cud/predicate.pickle","rb"))
        self.arg_pos = pickle.load(open("../../data/semantic-role-labeling/Cud/arguments.pickle","rb"))
        self.max_num_of_pred = max_num_of_pred
        self.model = model = model_builder(self.pos_pos, 
                                           self.lemma_pos,
                                           self.dep_rel_pos,
                                           self.pred_pos, 
                                           self.arg_pos,
                                           max_length = self.max_length,
                                           max_num_of_pred = self.max_num_of_pred
                                          )
        self.list_of_words = self.model.load_model("../../data/glove.6B.100d.txt")
        self.batch = 0
    
    def __call__(self, batch_size):
        self.batch = batch_size
        sents = self.reader.read_sentences(batch_size)
        X_emb, X_POS, X_LEMMA, Lapl_i, Lapl_v, Sp_i, Sp_v, Pred_Ind, Arg_Pos, Seq_Len = self.model.creation_for_SRL(sents)
        ind = np.random.choice(np.arange(X_emb.shape[0]), size=batch_size, replace=False)
           
        return X_emb[ind], X_POS[ind], X_LEMMA[ind], Lapl_i[ind], Lapl_v[ind], Sp_i[ind], Sp_v[ind], Pred_Ind[ind], Arg_Pos[ind], Seq_Len[ind]
    
    
    
class test_generator:
    def __init__(self, file, max_len_sent):
        self.file = file
        self.reader = read_file(self.file)
        self.pos_pos = pickle.load(open("../../data/semantic-role-labeling/Cud/part_of_speech.pickle","rb"))
        self.lemma_pos = pickle.load(open("../../data/semantic-role-labeling/Cud/lemma.pickle","rb"))
        self.dep_rel_pos = pickle.load(open("../../data/semantic-role-labeling/Cud/dependecy_relation.pickle","rb"))
        self.max_length = max_len_sent
        self.pred_pos = pickle.load(open("../../data/semantic-role-labeling/Cud/predicate.pickle","rb"))
        self.arg_pos = pickle.load(open("../../data/semantic-role-labeling/Cud/arguments.pickle","rb"))
        self.model = model = model_builder(self.pos_pos, 
                                           self.lemma_pos,
                                           self.dep_rel_pos,
                                           self.pred_pos, 
                                           self.arg_pos,
                                           max_length = self.max_length,
                                           max_num_of_pred = 26
                                          )
        self.list_of_words = self.model.load_model("../../data/glove.6B.100d.txt")
        self.batch = 0
        self.data = self.init()
        self.counter_sent = 200

    def init(self):
        sents_new = self.reader.read_sentences(200)
        return self.model.creation_for_test_srl(sents_new)
        
        
    def new_data(self, batch_size):
        sents_new = self.reader.read_sentences(batch_size)
        to_ret = []
        for D, D_n in zip(self.data, self.model.creation_for_test_srl(sents_new)):
            D_ret = np.concatenate([D,D_n], axis=0)
            to_ret.append(D_ret)

        return to_ret
    
    def give(self):
        to_give = []
        update = []
        for D in self.data:
            to_give.append(D[:self.batch])          
            D_update = D[self.batch:]
            update.append(D_update)
        self.data = update
        return to_give

    def __call__(self, batch_size):
        self.batch = batch_size
        
        if self.data[0].shape[0] >= batch_size:
            X_emb, X_POS, X_LEMMA, Lapl_i, Lapl_v, Sp_i,Sp_v, Pred_Ind, Seq_Len = self.give()
            
        else:
            self.data = self.new_data(batch_size)
            self.counter_sent += batch_size
            X_emb, X_POS, X_LEMMA, Lapl_i, Lapl_v, Sp_i,Sp_v, Pred_Ind, Seq_Len = self.give()
        
        
        
        return X_emb, X_POS, X_LEMMA, Lapl_i, Lapl_v, Sp_i,Sp_v, Pred_Ind, Seq_Len