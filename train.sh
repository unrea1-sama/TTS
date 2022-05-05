cd monotonic_align
python3 setup.py build_ext --inplace
cd ..
python3 train.py -c $1 -m $2