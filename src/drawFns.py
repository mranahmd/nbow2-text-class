import codecs

import numpy
import math
from tsne import bh_sne
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def getTsne(modelFile, outDir, NBOW2=True):
    pp = numpy.load(modelFile) 
    wv = pp['Wemb'].copy()

    sklearn_pca = PCA(n_components=50)
    Y_sklearn = sklearn_pca.fit_transform(wv)
    Y_sklearn = numpy.asfarray( Y_sklearn, dtype='float' )

    print "PCA transformation done ..."
    print "Waitig for t-SNE computation ..."
    
    reduced_vecs = bh_sne(Y_sklearn)

    with open(outDir + "/tsne", "w") as out:
        for i in range(len(reduced_vecs)):
            out.write(str(reduced_vecs[i,0]) + " " + str(reduced_vecs[i,1]) + "\n")
    out.close

    print "t-SNE written to file ..."
    
    if NBOW2:
        av = pp['AVs'].astype('float64').T[0]
        wts =[]
        for i in range(len(wv)):
            wt = sigmoid(numpy.dot(wv[i],av))
            wts.append(wt)
        with open(outDir + "/wts", "w") as out:
            for i in range(len(wts)):
                out.write(str(wts[i]) + "\n")
        out.close
    
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def printDoc(fname, lineNum, numWordsToShow=25):
	colorScale = ["#ffff00", 	#1.0
				"#ffcc66",
				"#ff6600",
				"#ff9966",
				"#ff0000",		#0.5
				"#ff6666",
				"#cc0099",
				"#9900cc",
				"#6600cc",
				"#333399"]		#0.0
	l = 0
	out = "<font size=\"5\" color=\"red\"> Not so many lines in file! </font>"
	f = codecs.open(fname, "r", "utf-8")
	for line in iter(f):
		l = l+1
		if l == lineNum:
			out = ""
			wcnt = 1
			for ww in line.split():
				wrd, wt = ww.split('_')
				wsize = int(float(wt) * 10) 
				if (wsize - float(wt)) != 0:
					wsize = wsize + 1
				wcolor = 10 - wsize
				wsize = wsize - 3
				if wsize < 1:
					wsize = 1
				currWrd = "<font size=\"" + str(wsize) +"\" color=\"" + colorScale[wcolor] + "\">" + wrd + "</font>"
				out = out + " " + currWrd
				if wcnt <= numWordsToShow:
					wcnt = wcnt + 1
				else:
					break
			break	
	f.close()
	return out + " <br> "

