import numpy
import sys

sidx = numpy.random.permutation(11293)
n_train = int(numpy.round(11293 * (0.85)))

data = []
file = open('20ng-train-all-terms.txt', 'r')
for line in file:  
    data.append(line)

i=0
for line in data:  
    if i <= n_train:
    	sys.stderr.write(data[sidx[i]])
    else:
	sys.stdout.write(data[sidx[i]])
    i=i+1
