config=$1
modeldir=$2
#python3 -m pip install cython
rm -rf monotonic_align/__pycache__ monotonic_align/build monotonic_align/monotonic_align monotonic_align/core.c
cd monotonic_align; python3 setup.py build_ext --inplace
cd ..
python3 init.py -c $config -m $modeldir
python3 train.py -c $config -m $modeldir
