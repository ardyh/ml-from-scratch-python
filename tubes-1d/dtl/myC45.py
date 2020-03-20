from .node import Node
import pandas as pd
import numpy as np
import math

#definisi kelas Tree
#Kelas ini mengkonstruksi decision tree dengan menghubungkan sekumpulan node, juga memilih untuk tiap node 
#atribut apa yang akan digunakan untuk splitting. Kelas ini dapat mempertimbangkan atribut yang mengandung nilai null.
#Metrik yang  digunakan bisa dipilih antara information gain atau gain ratio.
#Kelas ini dapat melakukan pruning pada tree yang dibuat, dan juga dapat mencetak model tree yang telah dibuat
#NOTE: Asumsi missing value, bernilai "None" atau "none"
#Atribut-atribut Tree:
# - data: merupakan data yang digunakan untuk training
# - target_attr: atribut yang menjadi target prediksi (label)
# - root: node yang merupakan root
# - use_info_gain: True/False. Jika true maka metrik pemilihan atribut menggunakan information gain. Jika False, metrik menggunakan gain ratio
class Tree:
    #konstruktor
    def __init__(self, data, target_attr, use_info_gain=True,root_value=None):
        self.data = data
        self.target_attr = target_attr
        self.root = None
        self.root_value = root_value
        self.use_info_gain = use_info_gain
        self.ruleset = []
        self.accuracy_ori = None
    
    #cari entropi total pada data
    def total_entropy(self, data):
        proportion = data[self.target_attr].value_counts()/len(data)
        entropy = 0
        for p in proportion.tolist():
            entropy -= p*math.log(p,2)
        return entropy
    
    #hitung information gain dari suatu kolom
    def info_gain(self, kolom):
        data = self.data
        data_entropy = self.total_entropy(data)
        proportion_kolom = data[kolom].value_counts()/len(data)
        sum_entropy_kolom = 0
        for value_kolom, value_proportion in zip(proportion_kolom.index.tolist(), proportion_kolom.tolist()):
            #print("here checking")
            #print(data[data[kolom] == value_kolom])
            entropy_value_kolom = self.total_entropy(data[data[kolom] == value_kolom])
            sum_entropy_kolom -= value_proportion*entropy_value_kolom
            
        return data_entropy + sum_entropy_kolom
    
    #hitung information split pada data di suatu atribut
    def split_info(self, attr):
        proportion = self.data[attr].value_counts()/len(data)
        split_info = 0
        for p in proportion.tolist():
            split_info -= p*math.log(p,2)
        return split_info
    
    #hitung gain ratio untuk suatu atribut
    def gain_ratio(self, attr):
        return info_gain(attr)/split_info(attr)
    
    #cari split-split yang memungkinkan pada atribut continuous
    def find_possible_splits_continuous(self, sorted_data, split_attr):
        sorted_target = sorted_data[self.target_attr].values.tolist()
        sorted_attr = sorted_data[split_attr].values.tolist()
        prev_target_value = sorted_target[0]
        possible_splits = []
        #iterasi target value, cari titik-titik dimana 
        try:
            for i in range(1, len(sorted_target)):
                el = sorted_target[i]
                if (prev_target_value != el):
                    possible_splits.append(0.5*(sorted_attr[i] + sorted_attr[i-1]))
                prev_target_value = el
        except Exception as e:
            print(e)
        finally:
            return possible_splits
    
    #cari gain dari tiap split dan cari split optimum
    def find_optimum_split_continuous(self, pos_splits, sorted_data, split_attr):
        optimum_split = 0
        max_info_gain = -1
        #iterate split
        for i, el in enumerate(pos_splits):
            #hitung information gain
            current_gain = self.calculate_info_gain_continuous(el, sorted_data, split_attr)
            #jika information gain lebih dari sebelumnya, ganti optimum split
            if(current_gain > max_info_gain):
                max_info_gain = current_gain
                optimum_split = el
        return optimum_split
    
    #cari information gain pada suatu split continuous
    def calculate_info_gain_continuous(self, split_value, sorted_data, split_attr):
        data_entropy = self.total_entropy(sorted_data)
        #pisah data mjd "<=" dan ">" split_value
        data_less_than_equal = sorted_data[sorted_data[split_attr] <= split_value]
        data_more_than = sorted_data[sorted_data[split_attr] > split_value]
        #hitung entropi kolom
        entropy_less_than_equal = (float(len(data_less_than_equal))/len(sorted_data)) * self.total_entropy(data_less_than_equal)
        entropy_more_than = (float(len(data_more_than))/len(sorted_data)) * self.total_entropy(data_more_than)
        return data_entropy - entropy_less_than_equal - entropy_more_than
    
    #check apakah attribute == numerik
    def is_attr_categorical(self, attr):
        return self.data[attr].dtype == 'O'
    
    #handling missing value
    def handle_missing_value(self, split_attr):
        if(self.is_attr_categorical(split_attr)):
            mode = self.data[split_attr].mode().values[0]
            self.data[split_attr] = self.data[split_attr].replace({None:mode})        
    
    #buat tree
    def make_tree(self):
        #cari info_gain dari masing-masing kolom 
        data_X = self.data.drop(self.target_attr, axis=1)
        
        #basis-1: jika data terbagi dg sempurna
        if(self.data[self.target_attr].nunique() == 1):
            self.root = Node("none", "none", self.target_attr, is_leaf=True, leaf_value=self.data[self.target_attr].unique()[0], parent_value=self.root_value)
            return self.root
        
        #basis-2: jika tidak ada atribut
        if(len(data_X.columns) == 0):
            self.root = Node("none", "none", self.target_attr, is_leaf=True, leaf_value=self.data[self.target_attr].mode().values[0], parent_value=self.root_value)
            return self.root
        
        #rekurens, jika data tidak bisa mjd leaf
        else:
            max_metric = -1
            split_attr = ""
            is_split_attr_categorical = True
            for attr in data_X.columns:
                #Jika kolom kategorikal
                if(self.is_attr_categorical(attr)):
                    if(self.use_info_gain):
                        current_metric = self.info_gain(attr)
                    else:
                        current_metric = self.gain_ratio(attr)
                #jika kolom numerik
                else:
                    #sort data
                    sorted_data = self.data.sort_values(by=attr)
                    #cari split-split yang memungkinkan 
                    pos_splits = self.find_possible_splits_continuous(sorted_data, attr)
                    #hitung gain dari tiap continuous split dan cari nilai optimum
                    split_value_continuous = self.find_optimum_split_continuous(pos_splits, sorted_data, attr)
                    #hitung gain ketika sudah diketahui nilai optimum
                    current_metric = self.calculate_info_gain_continuous(split_value_continuous, sorted_data, attr)

                #jika ditemukan maximum info gain di kolom tertentu
                if(current_metric > max_metric):
                    max_metric = current_metric
                    split_attr = attr
                    is_split_attr_categorical = self.is_attr_categorical(attr)
                    if (not is_split_attr_categorical):
                        split_value_attr = split_value_continuous
            
            #setelah atribut dipilih, cek apakah ada missing value
            #impute missing value dengan modus pada atribut tsb. (asumsi: atribut yg di handle hanyalah kategorikal)
            self.handle_missing_value(split_attr)
            
            #buat node
            #jika atribut terpilih == kategorikal
            if(is_split_attr_categorical):
                self.root = Node(self.data, split_attr, self.target_attr, parent_value=self.root_value)
                split_values = self.data[split_attr].unique()
                #iterate all split values
                for split_value in split_values:
                    filtered_data = self.data[self.data[split_attr] == split_value].drop(split_attr, axis=1)
                    self.root.add_child(Tree(filtered_data, self.target_attr, root_value=split_value).make_tree())

            #jika atribut terpilih == numerik & kontinu
            else:
                self.root = Node(self.data, split_attr, self.target_attr, is_continuous=True, split_value_continuous=split_value_attr, parent_value=self.root_value)
                #filter <=
                filtered_data = self.data[self.data[split_attr] <= split_value_attr].drop(split_attr, axis=1)
                self.root.add_child(Tree(filtered_data, self.target_attr, root_value="<="+str(split_value_attr)).make_tree())

                #filter >
                filtered_data = self.data[self.data[split_attr] > split_value_attr].drop(split_attr, axis=1)
                self.root.add_child(Tree(filtered_data, self.target_attr, root_value=">"+str(split_value_attr)).make_tree())

            return self.root

    def print_tree(self, node, depth, space):
        if (depth == 0):
            print('-------tree-------')
            dash = ''
        else:
            dash = '|' + '-'*space + '(' + node.parent_value + ')' + '-'*space + '>'
            
        if(node.is_leaf):
            output = ('|' + ('      '*space))*(depth-1) + dash + '{class : ' + str(node.leaf_value) + '}'
        else:
            output = ('|' + ('      '*space))*(depth-1) + dash + node.split_attr 
        
        if (node.parent_value):
            output = output 
        
        print(output)
        
        depth += 1
        for child in node.childs:
            self.print_tree(child, depth, space)
            
    #bagian rekursif untuk prediksi
    def get_prediction_result(self, prediction_instance, node):
        #basis - jika node merupakan leaf, kembalikan value
        if(node.is_leaf):
            return node.leaf_value
        
        #rekurens - jika node bukan leaf, cari anaknya yang tepat, telusuri anak
        else:
            #jika node categorical
            if(node.is_attr_categorical()):
                for child in node.childs:
                    if (child.parent_value == prediction_instance[node.split_attr]):
                        return self.get_prediction_result(prediction_instance, child)
                        break
            #jika node numerik/kontinu
            else:
                if(prediction_instance[node.split_attr] <= node.split_values[0]):
                    return self.get_prediction_result(prediction_instance, node.childs[0])
                elif(prediction_instance[node.split_attr] > node.split_values[0]):
                    return self.get_prediction_result(prediction_instance, node.childs[1])
                
    #prediksi suatu dataset test
    def predict(self, test_data):
        pred_result = []
        #iterasi seluruh instance pada test_data
        for i in range(len(test_data)):
            #instance untuk di prediksi
            prediction_instance = test_data.iloc[i]
            #get prediction untuk instance yang dicek, lalu append ke hasil
            pred_result.append(self.get_prediction_result(prediction_instance, self.root))
        return pred_result
    
    #transformasi tree menjadi kumpulan rule
    def recursively_write_rule(self, node, rule):
        #basis - mencapai leaf. Append rule ke ruleset
        if(node.is_leaf):
            new_rule = rule + [[self.target_attr, node.leaf_value]]
            self.ruleset.append(new_rule)
        
        #rekurens - mencatat current precondition dan telusuri anak-anaknya
        else:
            for child in node.childs:
                new_rule = rule + [[node.split_attr, child.parent_value]]
                self.recursively_write_rule(child, new_rule)
    
    #parsing rule menjadi query
    def parse_rule(self, rule):
        str_rule = ''
        for statement in rule[:-1]:
            #categorical variable
            if (statement[1][0] != "<") and (statement[1][0] != ">"):
                str_rule += statement[0] + ' == "' + statement[1] + '" and '
            else:
                str_rule += statement[0] + statement[1] + ' and '
        return str_rule[:-4]
    
    #kalkulasi akurasi suatu rule
    def calculate_rule_accuracy(self, rule):
        query = self.parse_rule(rule)
        filtered_data = self.data_test.query(query)
        target_value = rule[-1][1]
        num_correct_answers = len(filtered_data[filtered_data[self.target_attr] == target_value])
        if(len(filtered_data) == 0):
            return 0
        else:
            return float(num_correct_answers)/float(len(filtered_data))
    
    #pruning untuk suatu rule
    def prune_rule(self, rule, prev_accuracy):
        if(len(rule) > 2):
            optimal_rule = []
            max_accuracy = -1
            for statement in rule[:-1]:
                temp_rule = rule.copy()
                temp_rule.remove(statement)
                accuracy = self.calculate_rule_accuracy(temp_rule)
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    optimal_rule = temp_rule

            #basis - akurasi tidak improve
            if((max_accuracy <= prev_accuracy) or (len(rule) == 0)):
                return (optimal_rule, rule[-1][-1], max_accuracy)
            #rekurens - akurasi masih bisa dinaikkan dengan pruning
            else:
                return self.prune_rule(optimal_rule, max_accuracy)
        else:
            return (rule, rule[-1][-1], self.calculate_rule_accuracy(rule))
    
    #post-pruning
    def rule_post_pruning(self, data_test):
        print('-------rules-------')
        #definisikan rules yang ada
        #lakukan DFS pada tree sampai leaf. Catat semua rule yang ada
        self.recursively_write_rule(self.root, [])
        
        #set data test
        self.data_test = data_test
        
        #prune rule
        sorted_rule = {}
        for i, rule in enumerate(self.ruleset):
            pruned_rule, label, accuracy = (self.prune_rule(rule, -1))
            sorted_rule[self.parse_rule(pruned_rule)+'; label: '+str(label)] = accuracy

        sorted_rule = sorted(sorted_rule.items(), key=lambda kv: kv[1])
        return sorted_rule
    
    #rekurens accuracy_tree
    #return 0 jika salah
    #return 1 jika benar
    def check_result(self, check_instance, node):
        #basis - jika node merupakan leaf, kembalikan value
        if(node.is_leaf):
                return node.leaf_value == check_instance[node.target_attr]
        #rekurens - jika node bukan leaf, cari anak
        else:
            #jika node categorical
            if(node.is_attr_categorical()):
                for child in node.childs:
                    if (child.parent_value == check_instance[node.split_attr]):
                        return self.check_result(check_instance, child)
                        break
                        
            #jika node numerik/kontinu
            else:
                if(check_instance[node.split_attr] <= node.split_values[0]):
                    return self.check_result(check_instance, node.childs[0])
                elif(check_instance[node.split_attr] > node.split_values[0]):
                    return self.check_result(check_instance, node.childs[1])

    def accuracy_tree(self, test_data):
        success = 0;
        #iterasi seluruh instance pada test_data
        for i in range(len(test_data)):
            #instance untuk di prediksi
            success += self.check_result(test_data.iloc[i], self.root)
        return success/len(test_data)*100
    
    def pruning_rek(self, parent, node_child, test_data):
        if(parent==self.root):
            pass
        #basis jika node adalah leaf bandingkan akurasi pruned tree dan original tree
        elif(node_child.is_leaf):
            parent.is_leaf=True
            parent.leaf_value = parent.data[node_child.target_attr].mode().values[0]
            if(self.accuracy_tree(test_data)<self.accuracy_ori):
                parent.is_leaf=False
                parent.leaf_value = None
            else:
                parent.childs.clear()
                self.accuracy_ori = self.accuracy_tree(test_data)
        else :
            return self.pruning_rek(node_child,node_child.childs[0],test_data)    
            
    def post_pruning(self, data_val):
        self.accuracy_ori=self.accuracy_tree(data_val)
        for child in self.root.childs:
            if(not(child.is_leaf)):
                self.pruning_rek(child, child.childs[0], data_val)
