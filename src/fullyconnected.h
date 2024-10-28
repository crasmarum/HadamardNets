/*
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
*/

#ifndef FULLYCONNECTED_H_
#define FULLYCONNECTED_H_

#include "data.h"
#include "layer.h"

class FullyConn: public Layer {
	std::vector<Data*> filters_;
	std::vector<std::complex<float> > bias_;
	std::vector<std::complex<float> > dz_star_bias_;
public:
	FullyConn(int in_no_planes, int in_plane_size, int out_length) : Layer(in_no_planes, in_plane_size) {
		for (int var = 0; var < out_length; ++var) {
			filters_.push_back(new Data(1, input_.length_));
			std::fill(filters_[var]->real_, filters_[var]->real_ + filters_[var]->length_, 1.0);
			bias_.push_back(std::complex<float>(sqrt(2)/2, sqrt(2)/2));
			dz_star_bias_.push_back(0);
		}
	}

	virtual ~FullyConn() {
		for (int var = 0; var < filters_.size(); ++var) {
			delete filters_[var];
		}
	}

	virtual std::string getName() {
		return "FullyConn";
	}

	void getOutTemplate(int *no_planes, int *plane_size) {
		*no_planes = 1;
		*plane_size = filters_.size();
	}

	virtual void forward() {
		assert(output().length_ == filters_.size());
		for (int var = 0; var < output().length_; ++var) {
			float sum_re = 0.0;
			float sum_im = 0.0;
			Data& filter = *filters_[var];
			for (int pos = 0; pos < filter.length_; ++pos) {
				sum_re += (filter.real_[pos] * input_.real_[pos] + filter.imag_[pos] * input_.imag_[pos]);
				sum_im += (filter.real_[pos] * input_.imag_[pos] - filter.imag_[pos] * input_.real_[pos]);
			}
			output().real_[var] = sum_re + bias_[var].real();
			output().imag_[var] = sum_im + bias_[var].imag();
		}
	}

	// U : C^|input| -> C^|output|, U = (U1, ... U_j, ... U_|output()|), U_j : C^|input| -> C,
	// L : C^|output()| -> R
	// k = 1..|input|, j = 1..|output|
	// (d L(U) / dz_k) = sum_j (dL(U)/dz_j) (dU_j / dz_k) + sum_j (dL(U)/dz_j*) [(dU_j*/dz_k) == (dU_j / dz_k*)*]
	// (d L(U) / dz_k*) = sum_j (dL(U)/dz_j) (dU_j / dz_k*) + sum_j (dL(U)/dz_j*) [(dU_j*/dz_k*) == (dU_j / dz_k)*]
	virtual void backward(int label) {
		assert(output().length_ == filters_.size());

		for (int in_pos = 0; in_pos < input_.length_; ++in_pos) {
			std::complex<float> sum_dz = 0;
			std::complex<float> sum_dz_star = 0;

			for (int out_pos = 0; out_pos < output().length_; ++out_pos) {
				Data *filter = filters_[out_pos];
				std::complex<float> dLdzj(output().dz_real_[out_pos], output().dz_imag_[out_pos]);
				std::complex<float> dLdzj_star(output().dz_star_real_[out_pos], output().dz_star_imag_[out_pos]);
				std::complex<float> dU_jdz_k(filter->real_[in_pos], -filter->imag_[in_pos]);

				sum_dz += dLdzj * dU_jdz_k;
				sum_dz_star += dLdzj_star * std::conj(dU_jdz_k);
			}

			input_.dz_real_[in_pos] = sum_dz.real();
			input_.dz_imag_[in_pos] = sum_dz.imag();
			input_.dz_star_real_[in_pos] = sum_dz_star.real();
			input_.dz_star_imag_[in_pos] = sum_dz_star.imag();
		}

		for (int out_pos = 0; out_pos < output().length_; ++out_pos) {
			Data *filter = filters_[out_pos];
			std::complex<float> dLdzj(output().dz_real_[out_pos], output().dz_imag_[out_pos]);
			std::complex<float> dLdzj_star(output().dz_star_real_[out_pos], output().dz_star_imag_[out_pos]);

			for (int pos = 0; pos < filter->length_; ++pos) {
				std::complex<float> dU_jdz_star_k(input_.real_[pos], input_.imag_[pos]);
				// std::complex<float> sum_dz = dLdzj * dU_jdz_k;
				std::complex<float> sum_dz_star = dLdzj * dU_jdz_star_k;

				filter->dz_star_real_[pos] += sum_dz_star.real();
				filter->dz_star_imag_[pos] += sum_dz_star.imag();
			}
			dz_star_bias_[out_pos] += dLdzj_star;
		}
	}

	virtual void update(float learningRate) {
		for (int out_pos = 0; out_pos < output().length_; ++out_pos) {
			Data& filter = *filters_[out_pos];
			for (int pos = 0; pos < filter.length_; ++pos) {
				filter.real_[pos] -= learningRate * filter.dz_star_real_[pos];
				filter.imag_[pos] -= learningRate * filter.dz_star_imag_[pos];
				filter.dz_star_real_[pos] = 0.0;
				filter.dz_star_imag_[pos] = 0.0;
			}
			bias_[out_pos] -= learningRate * dz_star_bias_[out_pos];
			dz_star_bias_[out_pos] = 0.0;
		}
	}

	int getNoFilters() {
		return filters_.size();
	}

	Data* mutable_input() {
		return &input_;
	}

	Data* mutable_filter(int f_indx) {
		assert(f_indx >= 0 && f_indx < filters_.size());
		return filters_[f_indx];
	}

	std::complex<float>* mutable_bias(int f_indx) {
		assert(f_indx >= 0 && f_indx < filters_.size());
		return &(bias_[f_indx]);
	}

	std::complex<float> bias(int f_indx) {
		assert(f_indx >= 0 && f_indx < filters_.size());
		return bias_[f_indx];
	}

	std::complex<float> dz_star_bias(int f_indx) {
		assert(f_indx >= 0 && f_indx < filters_.size());
		return dz_star_bias_[f_indx];
	}
};

#endif /* FULLYCONNECTED_H_ */
