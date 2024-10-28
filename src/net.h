/*
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
*/

#ifndef NET_H_
#define NET_H_

#include <string>
#include <sstream>
#include <vector>
#include <initializer_list>

#include "activation.h"
#include "data.h"
#include "fullyconnected.h"
#include "hadamard.h"
#include "input.h"
#include "layer.h"
#include "model_utils.h"
#include "output.h"
#include "unityroots.h"



class no_planes {
public:
	int value_;
	no_planes(int value) {
		value_ = value;
	}
};

class no_filters {
public:
	int value_;
	no_filters(int value) {
		value_ = value;
	}

	static no_filters is(int value) {
		no_filters ret(value);
		return ret;
	}
};

class filter_shape {
	friend class shape_1D;
	friend class shape_2D;
	int dim1_;
	int dim2_;
	int dim_;
public:
	filter_shape(int dim1, int dim2, int dim)
		: dim1_(dim1), dim2_(dim2), dim_(dim) {
	}

	virtual ~filter_shape() {
	}
	virtual int getDim() {
		return dim_;
	}
	virtual int size() {
		return dim1_ * dim2_;
	}

	int check_kernel_size(int kernel_size) {
		if (kernel_size == dim1_) return dim2_;
		if (kernel_size == dim2_) return dim1_;
		assert(false);
		return 0;
	}
};

class shape_1D : public filter_shape {
public:
	shape_1D(int dim) : filter_shape(dim, 1, 1) {
	}
};

class shape_2D : public filter_shape {
public:
	shape_2D(int dim1, int dim2) : filter_shape(dim1, dim2, 2) {
	}
};

/*
 * A net is simply a list of layers.
 */
class CpuNet {
	std::vector<Layer*> net_;
	ModelSaver saver_;
	bool hasOutput_;
	int no_params_;

public:

	CpuNet() : hasOutput_(false), no_params_(0) {
	}

	virtual~CpuNet() {
		for (int var = 0; var < net_.size(); ++var) {
			delete net_[var];
		}
	}

	bool restore(std::string filePath) {
		assert(!net_.size());
		assert(saver_.Restore(net_, filePath));
		assert(net_.size());
		if (dynamic_cast<OutputLayer*>(net_.back())) {
			hasOutput_ = true;
		}
		no_params_ = 0;
		for (int l_indx = 0; l_indx < net_.size(); ++l_indx) {
			if (dynamic_cast<FullyConn*>(net_[l_indx])) {
				FullyConn *in = dynamic_cast<FullyConn*>(net_[l_indx]);
				no_params_ += in->getNoFilters() * (in->input().length_ + 1);
			}
			if (dynamic_cast<HadamardLayer*>(net_[l_indx])) {
				HadamardLayer *in = dynamic_cast<HadamardLayer*>(net_[l_indx]);
				no_params_ += in->getNoFilters() * (in->kernel_index()->size() + 1);
			}
		}
		checkArchitecture();
		return true;
	}

	CpuNet& addInputLayer(no_planes plane_count, int plane_size) {
		assert(!net_.size());
		net_.push_back(new InputLayer(plane_count.value_, plane_size));
		return *this;
	}

	CpuNet& addULayer() {
		assert(net_.size() && !hasOutput_);
		Layer *last = net_.back();
		assert(!dynamic_cast<ULayer*>(last) && !dynamic_cast<InputLayer*>(last));

		if (dynamic_cast<HadamardLayer*>(last)) {
			HadamardLayer *hadm = dynamic_cast<HadamardLayer*>(last);
			net_.push_back(new ULayer(hadm->getNoFilters(), last->input().plane_size_));
		} else if (dynamic_cast<FullyConn*>(last)) {
			FullyConn *fc = dynamic_cast<FullyConn*>(last);
			net_.push_back(new ULayer(1, fc->getNoFilters()));
		} else {
			assert(false);
		}

		last->setNextLayer(net_.back());
		return *this;
	}

	CpuNet& addHadamardLayer(no_filters filter_count, filter_shape shape, std::vector<int> kernel) {
		assert(net_.size() && !hasOutput_);
		Layer *last = net_.back();
		assert(dynamic_cast<ULayer*>(last) || dynamic_cast<InputLayer*>(last));
		assert(shape.getDim() && shape.size());
		assert(shape.getDim() && shape.size());
		assert(shape.getDim() != 2
			   || shape.size() <= last->input().plane_size_ * last->input().plane_size_); // FIX THIS
		assert(shape.getDim() != 1
			   || shape.size() <= last->input().plane_size_);

		if (shape.getDim() == 2) {
			shape.check_kernel_size(kernel.size());
			int length =  kernel.size();
			for (int i = 1; i < length; ++i) {
				for (int var = 0; var < shape.size() / length; ++var) {
					kernel.push_back(var + i * sqrt(last->input().plane_size_));
				}
			}
		}

		HadamardLayer *hadm = new HadamardLayer(last->input().no_planes_,
				                                last->input().plane_size_, kernel);
		net_.push_back(hadm);
		last->setNextLayer(hadm);
		// hadm->initFiltersRandomly(filter_count.value_, &rand_);
		no_params_ += (shape.size() + 1) * filter_count.value_;
		return *this;
	}

	CpuNet& addFullyConnLayer(no_filters filter_count) {
		assert(net_.size() && !hasOutput_);
		Layer *last = net_.back();
		//assert(dynamic_cast<ULayer*>(last) || dynamic_cast<InputLayer*>(last));

		FullyConn *fc = new FullyConn(last->input().no_planes_,
				                      last->input().plane_size_, filter_count.value_);
		net_.push_back(fc);
		last->setNextLayer(fc);
		//fc->initFiltersRandomly(&rand_);
		no_params_ += (last->input().length_ + 1) * filter_count.value_;
		return *this;
	}

	CpuNet& addOutLayer() {
		assert(net_.size() && !hasOutput_);
		Layer *last = net_.back();

		assert(dynamic_cast<FullyConn*>(last) || dynamic_cast<HadamardLayer*>(last));
		if (dynamic_cast<FullyConn*>(last)) {
			FullyConn *fc = dynamic_cast<FullyConn*>(last);
			net_.push_back(new OutputLayer(fc->getNoFilters()));
		} else if (dynamic_cast<HadamardLayer*>(last)) {
			HadamardLayer *hadm = dynamic_cast<HadamardLayer*>(last);
			assert(hadm->getNoFilters() == 1);
			net_.push_back(new OutputLayer(hadm->input().plane_size_));
		}

		last->setNextLayer(net_.back());
		hasOutput_ = true;
		return *this;
	}

	CpuNet& checkArchitecture() {
		assert(net_.size() && hasOutput_ && dynamic_cast<InputLayer*>(net_.front()));
		return *this;
	}

	void cpuForward() {
		for (int var = 0; var < net_.size(); ++var) {
			net_[var]->forward();
		}
	}

	void cpuForward(float *in_real, float *in_imag) {
		InputLayer *input = static_cast<InputLayer*>(net_[0]);
		input->forward(in_real, in_imag);
		for (int var = 1; var < net_.size(); ++var) {
			net_[var]->forward();
		}
	}

	void cpuBackward(int label) {
		for (int var = net_.size() - 1; var >= 0; --var) {
			net_[var]->backward(label);
		}
	}

	void cpuUpdate(float learning_rate) {
		for (int var = 0; var < net_.size(); ++var) {
			net_[var]->update(learning_rate);
		}
	}

	std::string toString() {
		std::ostringstream oss;
		oss << no_params_ << " params: ";
		for (int var = 0; var < net_.size(); ++var) {
			if (dynamic_cast<InputLayer*>(net_[var])) {
				InputLayer *in = dynamic_cast<InputLayer*>(net_[var]);
				oss << "(" << in->input().no_planes_ << " x " << in->input().plane_size_
		            << ") -> FFT -> ";
				continue;
			}
			if (dynamic_cast<OutputLayer*>(net_[var])) {
				OutputLayer *in = dynamic_cast<OutputLayer*>(net_[var]);
				oss << "(" << in->input().no_planes_ << " x " << in->input().plane_size_
					<< ") -> Out -> " << "("
					<< in->output().no_planes_ << " x " << in->output().plane_size_ << ")";
				continue;
			}
			if (dynamic_cast<ULayer*>(net_[var])) {
				ULayer *in = dynamic_cast<ULayer*>(net_[var]);
				oss << "(" << in->input().no_planes_ << " x " << in->input().plane_size_
						<< ") -> " << "U -> ";
				continue;
			}
			if (dynamic_cast<HadamardLayer*>(net_[var])) {
				HadamardLayer *in = dynamic_cast<HadamardLayer*>(net_[var]);
				oss << "(" << in->input().no_planes_ << " x " << in->input().plane_size_
					<< ") -> H(" << in->getNoFilters() << " x " << in->kernel_index()->size() << ") -> ";
				continue;
			}
			if (dynamic_cast<FullyConn*>(net_[var])) {
				FullyConn *in = dynamic_cast<FullyConn*>(net_[var]);
				oss << "(" << in->input().no_planes_ << " x " << in->input().plane_size_
				    << ") -> Fc("<< in->getNoFilters() << " x " << in->input().length_ << ") -> ";
				continue;
			}
		}
		return oss.str();
	}

	Layer* layer_at(int indx) {
		assert(indx >= 0 && indx < net_.size());
		return net_[indx];
	}

	Layer* cpu_back() {
		assert(net_.size());
		return net_.back();
	}

	Layer* cpu_front() {
		assert(net_.size());
		return net_.front();
	}
};

#endif /* NET_H_ */
