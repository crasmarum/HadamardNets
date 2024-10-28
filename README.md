# HadamardNets

ABOUT:

Code related to the "On the Equivalence of Convolutional and Hadamard Networks
using DFT" paper, see https://arxiv.org/abs/1810.11650. 
This software is a minimal proof of concept for showing that the complex Hadamard 
networks built with the specific layers presented in this paper are 
converging to a model.

 The main concepts from the corresponding paper are roughly mapped to code
 in the following way:
 
     1. A Hadamard net is a list of layers and it is implemented by the CpuNet
        class.
        
     2. All layers are instances of the Layer class. They need to support
        the forward and the backward propagation, see for example the
        FullyConn::forward() and the FullyConn::backward() methods.
        
     3. Tensors are simply arrays of complex numbers and they are implemented by the Data
        class. The Data class also keeps track of the Wirtinger gradients of the layer
        acting on its instances. For example, for the input tensor Z of the fully connected
        layer FC, (Z.dz_real_ + i * Z.dz_imag_) and (Z.dz_star_real_ + i * Z.dz_star_imag_)
        are the Wirtinger gradient and the conjugate Wirtinger gradient of FC with respect
        to Z and Z* = conjugate(Z), respectively.

BUILDING AND RUNNING THE CODE:

Assuming you have the GNU C++ compiler installed, you can build and
run the code as follows.

cd src/

g++ HadamardNet.cpp -o hadamard

./hadamard

If you run the program you should get this output:

Creating test data from data/t10k-images.idx3-ubyte...
read: 10000

Writing FFT(input) to the data/fft_10k_test.fft file.

	Note: will zero the first 5 frequencies.
 
wrote 1000 entries.

...

Loading model model/94percent_mnist_h7x7.mod...

789010 params: (1 x 784) -> FFT -> (1 x 784) -> H(100 x 49) -> (100 x 784) -> U -> (100 x 784) -> Fc(10 x 78400) -> (1 x 10) -> Out -> (1 x 10)

reading data from data/fft_10k_test.fft...

planes: 1 plane_size: 784

read: 10000 entries.

Computing model accuracy...

Current accuracy for 1000 entries: 0.935000

...

Final accuracy: 0.942800 = 9428.000000 / 10000

FILES:

README.txt
  This file.

data/t10k-images.idx3-ubyte
data/t10k-labels.idx1-ubyte
  These are the well known MNIST test data files.

model/94percent_mnist_h7x7.mod
  This is the 7x7 model obtained by training a shallow Hadamard network on the MNIST
  training data. Training took 64 epochs and used the Wirtinger gradient descent with a
  mini-batch of size 100.

HadamardNet.cpp

activation.h

data.h

fft_10k_test.fft - input, you can re-create it via command "cat fft_10k_test.fft.tgz_* | tar xz"

fullyconnected.h

hadamard.h

input.h

layer.h

model_utils.h

net.h

output.h

unityroots.h
  
  
  These are files that minimally implement the concepts presented in the
  "On the Equivalence of Convolutional and Hadamard Networks using DFT" paper.
  For example, the "output.h" file corresponds to the section
  "3.6. The Output Layer and the Cross Entropy Loss" in the corresponding paper.


