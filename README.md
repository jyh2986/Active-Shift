

# Active Shift Layer

This repository contains the implementation for Active Shift Layer (ASL).

Please see the paper [Constructing Fast Network through Deconstruction of Convolution](https://arxiv.org/abs/1806.07370). 

This paper is accepted in NIPS 2018 as spotlight session.

The code is based on [Caffe](https://github.com/BVLC/caffe)  
Tensorflow implementation is also available at [ASL-TF](https://github.com/jyh2986/Active-Shift-TF)


## Testing Code
You can validate backpropagation using test code.
Because it is not differentiable on lattice points, you should not use integer point position when you are testing code.
It is simply possible to define "TEST_ASHIFT_ENV" macro in <i>active_shift_layer.hpp</i>

1. Define "TEST_ASHIFT_ENV" macro in active_shift_layer.hpp
2. \> make test
3. \> ./build/test/test_active_shift_layer.testbin

You should pass all tests.
Before the start, <b>don't forget to undefine TEST_ASHIFT_ENV macro and make again.</b>



## Usage
ASL has 2 parameters : the shift amount (x,y) 

Using asl_param, you can control hyper-parameters for ASL. Please see the <i>caffe.proto</i>
