Cara penggunaan file
- Silakan run program MyMLP.py dengan command `python MyMLP.py`
- Dengan menjalankan program tersebut, dapat dilihat juga contoh penjalanan kelas MyMLP hingga ke tahap prediksi, dan sudah dibandingkan performanya dengan MLPClassifier Sklearn
- Bila ingin menggunakan kelas network, berikut penjelasan singkat dari metode-metode yang dapat digunakan:
  - Network()
    - n_inputs: jumlah input unit 
    - n_hidden: jumlah hidden unit
    - n_outputs: jumlah output unit
    - bias(default=1): nilai bias
    - learning_rate(default=0.1): nilai learning rate
  
  - fit() 
    - data: merupakan list of list yang berisi instances dari variable-variable prediktor
    - target: merupakan list 1d yang berisi nilai target untuk dilakukan training
    - epoch_limit(default=100): jumlah epoch maksimal
    - mini_batch_limit: ukuran mini batch
  
  - predict()
    -  data: merupakan list of list yang berisi instances dari variable-variable prediktor yang akan diprediksi