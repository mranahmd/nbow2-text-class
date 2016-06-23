#!/bin/bash

if [ $# -eq 3 ]; then
	kk=(0 1 2 3 4 5 6 7 8 9)
	for k in ${kk[@]}; do
		mod="$2/rt_nbow2_fold$k-model.npz"
		python -u prepareRtNbow2WtsSVMInput.py $1 $mod $3 $k 0 2>&1 | tee $3/rt_nbow2-wts_10k-cv_fold$k.out
		./liblinear-2.1/train -s 1 $3/rt_nbow2_fold$k.train.feats $3/rt_nbow2_fold$k.model-l2l2dual 2>&1 | tee $3/rt_nbow2_fold$k.svm-train.out
		./liblinear-2.1/predict $3/rt_nbow2_fold$k.test.feats $3/rt_nbow2_fold$k.model-l2l2dual $3/rt_nbow2_fold$k.model-l2l2dual.test-out 2>&1 | tee $3/rt_nbow2_fold$k.svm-test.out
		echo " "
	done
else
	echo "USAGE: $0 <input-dir> <model-dir> <output-dir>"
fi
