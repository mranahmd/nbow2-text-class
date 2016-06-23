from collections import OrderedDict
import cPickle as pkl
import sys, os
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import imdb_nbow2 as nbow2


def getWts(
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=50,  # The maximum number of epoch to run
    dispFreq=50,  # Display to stdout the training progress every N updates
    lrate=0.0001,  # Learning rate for initialization (not used for adadelta)
    optimizer=nbow2.adadelta,  # adadelta gradient descent training algorithm
    validFreq=780,  # Compute the validation error after this number of update.
    saveFreq=780,  # Save the parameters after every saveFreq updates
    maxlen=500,  # Sequence longer then this get ignored
    batch_size=32,  #16, # The batch size during training.
    test_batch_size=32,  # The batch size used for validation/test set.

    input_path = None,
    n_words=None,  # Vocabulary size
    dim_proj=300,  # word embeding dimension
    saveto='imdb_nbow2_model.npz',  # The best model will be saved there
    outDir=None,
    num_anch_vec=1,	# currently fixed to 1, but one can try more (experimental)

    random_init=1,	# word vector initialization, 0 -> randome or 1 -> glove word vectors in wordvectors file
    word_dropout=0.3,	# word dropout probability

    projFlag=0
):

    # Model options
    model_options = locals().copy()

    if input_path:
	wFile = input_path+'/vocab'
	vecFile = input_path+'/wordvectors'

	if os.path.isfile(wFile) :
		n_words = sum(1 for line in open(wFile))
		model_options['n_words'] = n_words
	else:
		print "Error: No worddict file ", wFile
		sys.exit()
	
	n_oovs = 2

	if os.path.isfile(vecFile) :
		ww = numpy.loadtxt(vecFile, dtype = numpy.float32)
		if ww.shape[1] != model_options['dim_proj']:
			print "Error: Mismatch in wordvectors dimension ", ww.shape[1], model_options['dim_proj']
			sys.exit()
	else:
		print "Error: No wordvectors file ", vecFile
		sys.exit()

    else:
	print "Error: Check input_path ", input_path
	sys.exit()

    if saveto:
	model= saveto
    else:
	print "Warning: outDir not specified. Will save outputs to current working directory..."
	sys.exit()

    print 'Loading data'
    data_x, data_y, = [], []
    with open(input_path + '/train-pos.id') as f:
	for line in f:
		bow=[]
		bow = [int(x) for x in line.split()]
        	data_x.append(bow)
		data_y.append(1)
    with open(input_path + '/train-neg.id') as f:
	for line in f:
		bow=[]
		bow = [int(x) for x in line.split()]
        	data_x.append(bow)
		data_y.append(0)
    train = (data_x, data_y)

    data_x, data_y = [], []
    with open(input_path + '/test-pos.id') as f:
	for line in f:
		bow=[]
		bow = [int(x) for x in line.split()]
        	data_x.append(bow)
		data_y.append(1)
    with open(input_path + '/test-neg.id') as f:
	for line in f:
		bow=[]
		bow = [int(x) for x in line.split()]
        	data_x.append(bow)
		data_y.append(0)
    test = (data_x, data_y)

    model_options['ydim'] = numpy.max(train[1]) + 1

    print 'Building model'
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = nbow2.init_params(model_options)

    nbow2.load_params(model, params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = nbow2.init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask, y, f_pred_prob, f_pred, cost, f_ops) = nbow2.build_model(tparams, model_options)
    use_noise.set_value(0.)

    print 'Calculating training features (wts) ...'
    kf = nbow2.get_minibatches_idx(len(train[0]), test_batch_size)
    trainFeats = open(outDir + '/train.feats', 'w')
    for _, valid_index in kf:
        x, mask, y = nbow2.prepare_data([train[0][t] for t in valid_index],
                                  numpy.array(train[1])[valid_index],
                                  maxlen=None)

	(aw, proj) = f_ops(x, mask)

	for i in range (0, len(y)):
		bow = numpy.zeros(n_words)
		doc = x[:, i].flatten()

		for j in range(0, x.shape[0]):
			bow[doc[j]] = bow[doc[j]] + aw[j][i]	
		bow = bow / numpy.sqrt((bow ** 2).sum())

		trainFeats.write(str(y[i]))
		for w in numpy.unique(doc):
			idx=w+1
			trainFeats.write(" " + str(idx) + ":" + str(bow[w]))

		if projFlag==1:
			idx = n_words
			for k in range(0, ww.shape[1]):
				idx=idx+1
				trainFeats.write(" " + str(idx) + ":" + str(proj[i][k]))

		trainFeats.write("\n") 
    trainFeats.close()

    print 'Calculating testing features (wts) ...'
    testFeats = open(outDir + '/test.feats', 'w')
    kf_test = nbow2.get_minibatches_idx(len(test[0]), test_batch_size)
    for _, valid_index in kf_test:
        x, mask, y = nbow2.prepare_data([test[0][t] for t in valid_index],
                                  numpy.array(test[1])[valid_index],
                                  maxlen=None)

	(aw, proj) = f_ops(x, mask)

	for i in range (0, len(y)):
		bow = numpy.zeros(n_words)
		doc = x[:, i].flatten()

		for j in range(0, x.shape[0]):
			bow[doc[j]] = bow[doc[j]] + aw[j][i]	
		bow = bow / numpy.sqrt((bow ** 2).sum())

		testFeats.write(str(y[i]))
		for w in numpy.unique(doc):
			idx=w+1
			testFeats.write(" " + str(idx) + ":" + str(bow[w]))

		if projFlag==1:
			idx = n_words
			for k in range(0, ww.shape[1]):
				idx=idx+1
				testFeats.write(" " + str(idx) + ":" + str(proj[i][k]))

		testFeats.write("\n") 
    testFeats.close()

    print 'SVM features (wts) ready ...'    

if __name__ == '__main__':
    if len(sys.argv) == 5:
	getWts(input_path=sys.argv[1], saveto=sys.argv[2], outDir=sys.argv[3], projFlag=int(sys.argv[4]))
    else:
	print "USAGE: ", sys.argv[0], "<input_dir>", "<model>", "<output_dir>", "<wtsNProj-flag>"
