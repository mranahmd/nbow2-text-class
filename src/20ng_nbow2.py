from collections import OrderedDict
import cPickle as pkl
import sys
import time
import os

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start: minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    """
    Global parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    if options['random_init'] == 1:
	randn = numpy.random.rand(options['n_words'], options['dim_proj'])
	params['Wemb'] = (0.01 * randn).astype(config.floatX)

    params['AVs'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                              options['num_anch_vec']).astype(config.floatX)

    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

def adadelta(lr, tparams, grads, x, mask, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update

def in_dropout_layer(use_noise, trng, mask, options):  
    noiseRet = tensor.switch(use_noise,
                         trng.binomial((mask.shape[0], mask.shape[1]),
                                        p=(1-options['word_dropout']), n=1,
                                        dtype=mask.dtype),
                         mask)

    return noiseRet

def build_model(tparams, options):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])

    noise = in_dropout_layer(use_noise, trng, mask, options)
    mask2 = mask * noise
    den = mask2.sum(axis=0)
    AllMaskId = tensor.switch(tensor.eq(den, 0), 0, 1) 
    finalMask = mask2 * AllMaskId + mask * (1 - AllMaskId)
    embMasked = emb * finalMask[:, :, None]

    wWts = tensor.dot(embMasked, tparams['AVs']).dimshuffle(1, 2, 0).reshape([n_samples * options['num_anch_vec'], n_timesteps])

    wWtsNorm = tensor.nnet.sigmoid(wWts).reshape([n_samples, 
                                                  options['num_anch_vec'],
						  n_timesteps])

    wWtsTens = wWtsNorm.dimshuffle(2, 0, 1)
    wWtsAvg = wWtsTens.sum(2) / options['num_anch_vec']

    attProj = embMasked * wWtsAvg[:, :, None]
    attProj = attProj.sum(axis=0) / finalMask.sum(axis=0)[:, None]

    pred = tensor.nnet.softmax(tensor.dot(attProj, tparams['U']) + tparams['b'])

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    NLL = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

    # symbolic Theano variable that represents the squared L2 term
    L2_sqr = tensor.sum(tparams['AVs'] ** 2) + tensor.sum(tparams['U'] ** 2) + tensor.sum(tparams['b'] ** 2) + tensor.sum(embMasked ** 2)

    lamda2 = 1e-5

    # the loss
    cost = NLL + lamda2 * L2_sqr

    f_pred_prob = theano.function([x, mask], [pred, wWtsTens], name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')
    f_ops = theano.function([x, mask], [wWtsAvg * finalMask, attProj], name='f_ops')

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost, f_ops


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x, mask)[0]
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print '%d/%d samples classified' % (n_done, n_samples)

    return probs


def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        preds = f_pred(x, mask)
        targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err

def get_ranks(f_pred_prob, prepare_data, data, iterator, options, outPrefix, verbose=False):
    valid_err = 0
    sys.stderr.write('\n------------------------------------------------------------------\n')

    pnDict = []
    with open(options['input_path'] + "/labeldict", "r") as ins:
	for line in ins:
		w = line.strip()
		pnDict.append(w)

    wDict = []
    with open(options['input_path'] + "/worddict", "r") as ins:
	for line in ins:
		w = line.strip()
		wDict.append(w)

    outLabels = open(outPrefix + 'oov-labels', 'w')
    outRanks = open(outPrefix + 'oov-ranks', 'w')
    wNwtFile = open(outPrefix + 'wNwts', 'w')
    wtFile = open(outPrefix + 'wts', 'w')
    tiFile = open(outPrefix + 'ti', 'w')

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)

	numpy.savetxt(outLabels, y, fmt='%d')
	
	ret = f_pred_prob(x, mask)
	pred_probs = ret[0]
	wts = ret[1]
        preds = pred_probs.argmax(axis=1)
        targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()

    	sortedPNIndexes = numpy.argsort(pred_probs, axis=1)
	for i in range (0, len(y)):
		sys.stderr.write(str(y[i]) + ' ' + pnDict[y[i]] + ' : ')

		for k in range(0, options['num_anch_vec']):
			for j in range(0, wts.shape[0]):
				if mask[j][i] == 1:
					if k == 0:
						wtFile.write(str(wts[j][i][k]) +' ')
						tiFile.write(str(x[j][i]) +' ')
						wNwtFile.write(wDict[x[j][i]] + '_' + str(wts[j][i][k]) +' ')
		wNwtFile.write('\n')
		wtFile.write('\n')
		tiFile.write('\n')

		for j in reversed(range(0, len(sortedPNIndexes[i]))):
			if(sortedPNIndexes[i][j] == y[i]):
				rank = len(sortedPNIndexes[i]) - j
				sys.stderr.write(str(rank) + '\n')
				outRanks.write(str(rank)+'\n')
				break

    wtFile.close
    tiFile.close
    outLabels.close
    outRanks.close
    wNwtFile.close
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err


def prepare_data(seqs, labels, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y, z in zip(lengths, seqs, labels, labels2):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.
       
    return x, x_mask, labels


def train_mlp(
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=300,  # The maximum number of epoch to run
    dispFreq=50,  # Display to stdout the training progress every N updates
    lrate=0.0001,   # Learning rate initialization (not used for adadelta)
    optimizer=adadelta,  # adadelta gradient descent training algorithm
    validFreq=300,  # Compute the validation error after this number of update.
    saveFreq=300,  # Save the parameters after every saveFreq updates
    maxlen=500,  # Sequence longer then this get ignored
    batch_size=32,  #16, # The batch size during training.
    test_batch_size=32,  # The batch size used for validation/test set.

    input_path = None,
    n_words=None,  # Vocabulary size
    dim_proj=300,  # word embeding dimension
    num_anch_vec=1,	# currently fixed to 1, but one can try more (experimental)
    saveto='20ng_nbow2_model.npz',  # The best model will be saved there
    outDir=None,

    random_init=1,	# word vector initialization, 0 -> randome or 1 -> skip-gram word vectors in wordvectors file
    word_dropout=0.75	# word dropout probability
):

    # Model options
    model_options = locals().copy()

    print "model options", model_options

    if input_path:
	wFile = input_path+'/worddict'
	oFile = input_path+'/labeldict'
	vecFile = input_path+'/wordvectors'

	if os.path.isfile(wFile) :
		n_words = sum(1 for line in open(wFile))
		model_options['n_words'] = n_words
	else:
		print "Error: No worddict file ", wFile
		sys.exit()
	
	if os.path.isfile(oFile) :
		n_oovs = sum(1 for line in open(oFile))
	else:
		print "Error: No labeldict file ", oFile
		sys.exit()

	if random_init== 0:
		if os.path.isfile(vecFile) :
			w = numpy.loadtxt(vecFile, dtype = numpy.float32)
			if w.shape[1] != model_options['dim_proj']:
				print "Error: Mismatch in wordvectors dimension ", w.shape[1], model_options['dim_proj']
				sys.exit()
		else:
			print "Error: No wordvectors file ", vecFile
			sys.exit()

    else:
	print "Error: Check input_path ", input_path
	sys.exit()

    if outDir:
	saveto= outDir+ "/20ng_nbow2_model.npz"
    else:
	print "Warning: outDir not specified. Will save outputs to current working directory..."
	outDir = "./"

    print 'Loading data'
    data_x, data_y, = [], []
    with open(input_path + '/trainX.id') as f:
        data_x = [[int(x) for x in line.split()] for line in f]
    with open(input_path + '/trainY.id') as f:
        data_y = [int(x) for x in f]
    train = (data_x, data_y)

    data_x, data_y = [], []
    with open(input_path + '/devX.id') as f:
        data_x = [[int(x) for x in line.split()] for line in f]
    with open(input_path + '/devY.id') as f:
        data_y = [int(x) for x in f]
    dev = (data_x, data_y)

    data_x, data_y = [], []
    with open(input_path + '/testX.id') as f:
        data_x = [[int(x) for x in line.split()] for line in f]
    with open(input_path + '/testY.id') as f:
        data_y = [int(x) for x in f]
    test = (data_x, data_y)

    print "%d train examples" % len(train[0])
    print "%d valid examples" % len(dev[0])
    print "%d test examples" % len(test[0])

    ydim = numpy.max(train[1]) + 1
    if ydim != n_oovs:
	print "Error: Mismatch in num of oovs ", ydim, n_oovs
	sys.exit()

    model_options['ydim'] = ydim

    print 'Building model'
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)

    if random_init == 0:
	print 'Initializing word vectors ...'
	params['Wemb'] = w
  
    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask, y, f_pred_prob, f_pred, cost, f_ops) = build_model(tparams, model_options)

    grads = tensor.grad(cost, wrt=tparams.values())
    f_grad = theano.function([x, mask, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads, x, mask, y, cost)

    print 'Optimization'

    kf_dev = get_minibatches_idx(len(dev[0]), test_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), test_batch_size)

    use_noise.set_value(0.)
    dev_err = pred_error(f_pred, prepare_data, dev, kf_dev)
    test_err = pred_error(f_pred, prepare_data, test, kf_test)
    print ('Staring Errors: Dev ', dev_err, 'Test ', test_err)

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) / batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in xrange(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t] for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = prepare_data(x, y)
                n_samples += x.shape[1]

                cost = f_grad_shared(x, mask, y)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print 'Saving...',

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print 'Done'

                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    train_err = pred_error(f_pred, prepare_data, train, kf)
                    dev_err = pred_error(f_pred, prepare_data, dev, kf_dev)
                    test_err = pred_error(f_pred, prepare_data, test, kf_test)
                    history_errs.append([dev_err, test_err])

                    if (uidx == 0 or
                        dev_err <= numpy.array(history_errs)[:, 0].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0

                    print ('Train ', train_err, 'Dev ', dev_err, 'Test ', test_err)

                    if (len(history_errs) > patience and dev_err >= numpy.array(history_errs)[:-patience, 0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

            print 'Seen %d samples' % n_samples

            if estop:
                break

    except KeyboardInterrupt:
        print "Training interupted"

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
    dev_err = pred_error(f_pred, prepare_data, dev, kf_dev)
    test_err = pred_error(f_pred, prepare_data, test, kf_test)
    tmp = get_ranks(f_pred_prob, prepare_data, test, kf_test, model_options, outDir+"/test-")
    print 'Train ', train_err, 'Dev ', dev_err, 'Test ', test_err

    if saveto:
        numpy.savez(saveto, train_err=train_err,
                    dev_err=dev_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print 'The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
    print >> sys.stderr, ('Training took %.1fs' %
                          (end_time - start_time))
    return train_err, dev_err, test_err


if __name__ == '__main__':
    if len(sys.argv) == 3:
	train_mlp(max_epochs=300, input_path=sys.argv[1], outDir=sys.argv[2])
    else:
	print "USAGE: ", sys.argv[0], "<input_dir>", "<output_dir>"
