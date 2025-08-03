#!/usr/bin/env bash
#/bin/bash

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_CFLAGS=( $(python -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))") )
TF_LFLAGS=( $(python -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))") )

# TF1.4
# /usr/local/cuda-10.1/bin/nvcc tf_approxmatch_g.cu -o tf_approxmatch_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
nvcc tf_approxmatch_g.cu -o tf_approxmatch_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++17 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -O2 -I/usr/local/cuda-11.2/include -L/usr/local/cuda-11.2/lib64 ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -lcudart -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework