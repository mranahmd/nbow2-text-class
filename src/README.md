A quick description of the main source files and their dependencies are give below.

[20ng_nbow.py]
- Train and test the NBOW-RAND model on the 20 newsgroup topic classification task. To use pre-trained vectors download glove.42B.300d, use script in ‘data/20ng’ directory to extract relevant word vectors and set ‘random_init=0’ in 20ng_nbow.py  

[20ng_nbow2.py]
- Train and test the NBOW2-RAND model on the 20 newsgroup topic classification task. To use pre-trained vectors download glove.42B.300d, use script in ‘data/20ng’ directory to extract relevant word vectors and set ‘random_init=0’ in 20ng_nbow2.py

[imdb_nbow2.py]
- Train and test the NBOW2-RAND model on the IMDB binary classification task. To use pre-trained vectors download glove.42B.300d, use script in ‘data/20ng’ directory to extract relevant word vectors and set ‘random_init=0’ in 20ng_nbow2.py

[rt_nbow2.py]
- Train and test the NBOW2-RAND model on 1 fold of the RT binary classification task.
To perform the 10-fold train-test use the ‘run_rt_nbow2_10k-cv.sh’ batch script. To use pre-trained vectors download glove.42B.300d, use script in ‘data/20ng’ directory to extract relevant word vectors and set ‘random_init=0’ in 20ng_nbow2.py

[prepareImdbNbow2WtsSVMInput.py]
- Prepares the liblinear/libsvm supported BOW format for the SVM classification experiment on the IMDB NBOW2 model word importance weights presented in Table 3 of the paper.

[prepareRtNbow2WtsSVMInput.py]
- Prepares the liblinear/libsvm supported BOW format for the SVM classification experiment on the RT NBOW2 model word importance weights presented in Table 3 of the paper. Again the batch script ‘run_rt_nbow2-wts_10k-cv.sh’ can be directly used for the 10 fold cross validation.

[drawFns.py]
- Function for drawing/writing documents with word importance information. Used for visualisation purpose in rt_nbow2_visualization.ipynb

[rt_nbow2_visualization.ipynb]
- A quick overview of the NBOW model when applied on the RT task
