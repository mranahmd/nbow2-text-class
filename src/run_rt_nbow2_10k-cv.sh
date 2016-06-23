#!/bin/bash

if [ $# -eq 2 ]; then
	kk=(0 1 2 3 4 5 6 7 8 9)
	for k in ${kk[@]}; do
		python -u rt_nbow2.py $1 $2 $k 2>&1 | tee $2/rt_nbow2_10k-cv_fold$k.out
		echo " "
	done
else
	echo "USAGE: $0 <input-dir> <output-dir>"
fi
