#!/usr/bin/env bash

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_CFLAGS=( $(python -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))") )
TF_LFLAGS=( $(python -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))") )

nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++17 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -O2 -I/usr/local/cuda-11.2/include -L/usr/local/cuda-11.2/lib64 ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -lcudart -I$TF_INC/external/nsync/public -L$TF_LIB