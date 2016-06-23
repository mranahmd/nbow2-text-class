# nbow2-text-class

This repository contains a python implementation of the NBOW2 model (**N**eural **B**ag of **W**eighted **W**ords), which enables a neural bag-of-words model to learn task specific importance of words. It also contains the scripts to reproduce the clasification results on the IMDB and RT Sentiment classification tasks and the 20-Newsgroup topic classification task.

These results were described in the RepL4NLP ACL 2016 paper:

> *Learning Word Importance with the Neural Bag-of-Words Model* by Sheikh Imran, Irina Illina, Dominique Fohr and Georges Linares.

For a quick overview take a look at the [ipython notebook on RT task](https://github.com/mranahmd/nbow2-text-class/edit/master/src/rt_nbow2_visualization.ipynb). Otherwise you can follow the instructions available below and the source code description available in the README in `src` directory.

---

## Instructions
The python scripts in the `src` directory will train and evaluate the performance of the NBOW2 model on the Sentiment and topic classification tasks presented in the paper. 

For simplicity and re-usability, the python scripts reuse most of the code of the [MLP example from the Theano website](http://deeplearning.net/tutorial/mlp.html). (Although this leads to a computationally slower version and a faster version can be implemented in python or C.)

In order to run these scripts, and reproduce the results in the paper, the following pre-requisites must be fulfilled:
- Python 2.7 or higher, Theano 0.7 or higher 
- This code can train the NBOW2-RAND model mentioned in the paper on IMDB, RT and 20ng tasks by using the corresponding data files in the ‘data’ directory accompanying this source code.
- To run the NBOW2 experiments which use pre-trained glove word vector initialisation you must download the glove.42B.300d word vectors from the [GloVe webpage](http://nlp.stanford.edu/projects/glove/). Then the scripts in the `data` directory in this repository can be used to extract word vectors used in the respective tasks.
- [LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) will be required to run the SVM based experiments on NBOW2 weights, as presented in Table 1 of the paper. (liblinear-2.1 was used in original experiments.)

Relevant notes:
- The experiments reported in the paper were performed on a Ubuntu machine with NVIDIA Quadro K4000 GPU and with THEANO_FLAGS=‘floatX=float32’
- The `models` directory stores the outputs and results of the training and testing performed with this code (and mentioned in the paper). The authors were able to reproduce these outputs (with differences after the 7th decimal place) while running on a Apple laptop computer.
- The `data` directory with the source stores term index and vocabulary files for each of the task. The scripts used to obtain these are also provided. The original datasets may or may not be provided but atleast the files used from the original datasets are hinted.
