/*
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
*/

#ifndef LAYER_H_
#define LAYER_H_

#include "data.h"

class Layer {
	friend class ULayer;
	friend class InputLayer;
	friend class OutputLayer;
	friend class HadamardLayer;
	friend class FilterLayer;
	friend class BiasLayer;
	friend class FullyConn;

protected:
	Data input_;
	Layer* next_;  // not owned
	bool is_divide_layer_;

public:
	Layer(int no_planes, int plane_size) :
			input_(no_planes, plane_size), next_(0), is_divide_layer_(false) {
	}

	virtual ~Layer() {
	}

	virtual void copyInputFrom(const float* input_re, const float* input_im) {
		std::copy(input_re, input_re + input_.length_, input_.real_);
		std::copy(input_im, input_im + input_.length_, input_.imag_);
	}

	void setNextLayer(Layer* next) {
		assert(!next_ && next);
		//assert(next->input_.plane_size_ == input_.plane_size_);
		if (is_divide_layer_) {
			assert(input_.plane_size_ % 2 == 0);
			next->mutable_input()->no_planes_ = input_.no_planes_ * 2;
			next->mutable_input()->plane_size_ = input_.plane_size_ / 2;
		}

		next_ = next;
	}

	const Data& input() const {
		return input_;
	}

	virtual inline const Data& output() {
		assert(next_);
		return next_->input_;
	}

public:
	virtual void getOutTemplate(int *no_planes, int *plane_size) = 0;
	virtual std::string getName() = 0;
	virtual void forward() = 0;
	virtual void backward(int label) = 0;
	virtual void update(float learningRate) = 0;
	virtual void updateInput(float learningRate) {
		for (int pos = 0; pos < input_.length_; ++pos) {
			input_.real_[pos] -= learningRate * input_.dz_star_real_[pos];
			input_.imag_[pos] -= learningRate * input_.dz_star_imag_[pos];
		}
	}

	Data* mutable_input() {
		return &input_;
	}

	const Layer* next() {
		return next_;
	}
};

#endif /* LAYER_H_ */
