Silakan melihat source code pada folder src. Terdapat beberapa file:
1. main.py
	Merupakan driver testing dari algoritma tree. Silakan dirun dengan komando python main.py atau python3 main.py
	untuk melihat performa algoritma C45 dan ID3 printing tree, predict, pruning menggunakan reduced-error 
	post-pruning dan rule post-pruning pada dataset iris dan play-tennis. 

2. myC45.py
	Merupakan algoritma decision tree C4.5 Untuk menggunakan modul ini, lakukan instruksi sbb:
	- Instansiasi tree dengan format
		tree = myC45(full_data, target_attribute)
	kemudian dapat dipanggil metode-metode sbb:
	- untuk membuat decision tree, panggil make_tree()
	- untuk print tree, panggil metode print_tree()
	- untuk pruning, panggil metode rule_post_pruning(testing_data) atau post_pruning(testing_data) (Harus dilakukan make_tree terlebih dahulu)
	- untuk predict, panggil metode predict(testing_data)
	

3. myID3.py
	Merupakan algoritma decision tree C4.5 Untuk menggunakan modul ini, lakukan instruksi sbb:
	- Instansiasi tree dengan format
		tree = myID3(full_data, target_attribute)
	kemudian dapat dipanggil metode-metode sbb:
	- untuk membuat decision tree, panggil make_tree()
	- untuk print tree, panggil metode print_tree()
	- untuk predict, panggil metode predict(testing_data)

4. node.py
	Hanya merupakan module pembantu untuk myID3 dan myC45