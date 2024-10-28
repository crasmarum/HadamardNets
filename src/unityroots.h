/*
Copyright (c) 2019 Anonymous@ICML2019

Additional material for the "On the Equivalence of Convolutional and Hadamard Networks
using DFT" paper.

This code should not be made public at this time.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
*/

#ifndef UNITYROOTS_H_
#define UNITYROOTS_H_

#include <cmath>
#include <vector>
#include <complex>
#include <assert.h>

class UnityRoots {
	std::vector<std::complex<float> > root_;

public:
	UnityRoots(int N) {
		for (int var = 0; var <= N - 1; ++var) {
			std::complex<float> r(std::cos(-2 * var * M_PI / N), std::sin(-2 * var * M_PI / N));
			root_.push_back(r);
		}
	}

	std::complex<float> root(int indx) {
		assert(indx >= 0);
		return root_[indx % root_.size()];
	}

	int size() {
		return root_.size();
	}
};

#endif /* UNITYROOTS_H_ */
