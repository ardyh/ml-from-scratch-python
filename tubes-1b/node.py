#definisi kelas Node
#Node merupakan split point pada tree. 
#Kelas ini menyimpan data yang ada pada suatu split point, atribut apa yang digunakan untuk splitting, dan tipe atribut tsb. Atau jika node merupakan daun maka disimpan value-nya
#Kelas ini dapat menentukan splitting point kebawah dari suatu node, baik atribut splittingnya kontinu maupun diskrit
#Atribut-atribut Node: 
# - data: subset data
# - split_attr: nama atribut yang akan di split
# - split_values: value cabang dari node (merupakan satu integer jika continuous, dan multiple values jika categorical)
# - target_attr: atribut label/atribut target prediksi
# - attr_cont_split: splitting point dari atribut tsb (jika atribut tsb kontinu)
# - is_leaf: apakah node merupakan daun atau tidak
# - leaf_value: nilai hasil prediksi jika node merupakan daun
# - childs: anak dari node yang berupa node
class Node:
    #konstruktor
    def __init__(self, data, split_attr, target_attr, is_continuous=False, split_value_continuous=None, is_leaf=False, leaf_value=None, parent_value=None):
        self.data = data
        self.split_attr = split_attr
        self.target_attr = target_attr
        self.childs = []
        self.is_leaf = is_leaf
        self.split_values = [split_value_continuous]
        self.leaf_value = leaf_value
        self.parent_value = parent_value

    #check apakah split attribute == numerik
    def is_attr_categorical(self):
        return self.data[self.split_attr].dtype == 'O'
    
    #get splits node jika node bukan daun
    def get_splits(self):
        if( not self.check_if_leaf()):
            #jika atribut split categorical
            if(self.is_attr_categorical()):
                #tentukan split values
                self.split_values = self.data[self.split_attr].unique()
            #jika atribut numerik / continuous, split value sudah didefinisikan sejak konstruksi objek
            return self.split_values
                        
    #add a child to a node
    def add_child(self, node):
        self.childs.append(node)