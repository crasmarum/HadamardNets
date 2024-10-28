/*
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
*/

#ifndef HADAMARD_H_
#define HADAMARD_H_

#include "data.h"
#include "layer.h"
#include "unityroots.h"

class FilterLayer: public Layer {
	friend class HadamardLayer;
	static FastFourierTransform<float> fft_;  // thread safe
	Data data_;
	const std::vector<int> *index_;
	std::complex<float> bias_;
	std::complex<float> dz_bias_;
	std::complex<float> dz_star_bias_;

public:

	FilterLayer(const std::vector<int> *kernel_index, Layer *parent)
			: Layer(parent->input().no_planes_, kernel_index->size()),
			  data_(parent->input().no_planes_, parent->input().plane_size_) {
		index_ = kernel_index;
	}

	FilterLayer(const FilterLayer& other) :
			Layer(other.input_.no_planes_, other.input_.plane_size_), data_(
					other.data_.no_planes_, other.data_.plane_size_) {
		const Data* source[2] = { &other.input_, &other.data_ };
		Data* dest[2] = { &input_, &data_ };
		for (int indx = 0; indx < 2; ++indx) {
			std::copy(source[indx]->real_,
					source[indx]->real_ + 6 * source[indx]->length_,
					dest[indx]->real_);
		}
		index_ = other.index_;
		bias_ = other.bias_;
		dz_bias_ = other.dz_bias_;
		dz_star_bias_ = other.dz_star_bias_;
	}

	virtual std::string getName() {
		return "FilterLayer";
	}

	void getOutTemplate(int *no_planes, int *plane_size) {
	}

	virtual ~FilterLayer() {
	}

	virtual void backward(int label) {
	}

	virtual void update(float learningRate) {
	}

	virtual void forward() {
		std::fill(data_.real_, data_.real_ + data_.length_, 0.0);
		std::fill(data_.imag_, data_.imag_ + data_.length_, 0.0);

		FOREACH_PLANE(plane, offset, data_) {
			int in_offset = plane * input_.plane_size_;
			for (int var = 0; var < input_.plane_size_; ++var) {
				data_.real_[offset + index_->at(var)] = input_.real_[in_offset + var];
				data_.imag_[offset + index_->at(var)] = input_.imag_[in_offset + var];
			}
			fft_.transform(data_.plane_size_, data_.real_ + offset, data_.imag_ + offset);
		}
	}

	const Data& data() const {
		return data_;
	}

	void setBias(std::complex<float> bias) {
		bias_ = bias;
	}

	const std::complex<float> bias() const {
		return bias_;
	}

	const std::complex<float> dz_star_bias() const {
		return dz_star_bias_;
	}
};

class HadamardLayer: public Layer {
	std::vector<FilterLayer> filters_;
	UnityRoots u_root_;
	std::vector<int> kernel_index_;
public:
	HadamardLayer(int no_planes, int plane_size, std::vector<int> kernel_index) :
			Layer(no_planes, plane_size), u_root_(plane_size) {
		kernel_index_ = kernel_index;
	}

	HadamardLayer(int no_planes, int plane_size) :
				Layer(no_planes, plane_size), u_root_(plane_size) {
		for (int var = 0; var < plane_size; ++var) {
			kernel_index_.push_back(var);
		}
	}

	HadamardLayer(int no_planes, int plane_size, int kernel_size) :
				Layer(no_planes, plane_size), u_root_(plane_size) {
		for (int var = 0; var < kernel_size; ++var) {
			kernel_index_.push_back(var);
		}
	}

	virtual ~HadamardLayer() {
	}

	virtual std::string getName() {
		return "HadamardLayer";
	}

	void getOutTemplate(int *no_planes, int *plane_size) {
		*no_planes = filters_.size();
		*plane_size = input_.plane_size_;
	}

	virtual void forward() {
		assert(next_ && filters_.size() == output().no_planes_);
		assert(filters_[0].data_.plane_size_ == input_.plane_size_);
		assert(filters_[0].data_.no_planes_ == input_.no_planes_);
		assert(input_.plane_size_ == output().plane_size_);


		FOREACH_PLANE(out_plane, out_offset, output())
		{
			FilterLayer& filter = filters_[out_plane];

			for (int pos = 0; pos < output().plane_size_; ++pos) {
				float sum_re = 0.0;
				float sum_im = 0.0;
				FOREACH_PLANE(plane, offset, input_) {
					float re1 = input_.real_[offset + pos];
					float im1 = input_.imag_[offset + pos];
					float re2 = filter.data_.real_[offset + pos];
					float im2 = filter.data_.imag_[offset + pos];
					sum_re += (re1 * re2 - im1 * im2);
					sum_im += (re1 * im2 + im1 * re2);
				}

				output().real_[out_offset + pos] = sum_re;
				output().imag_[out_offset + pos] = sum_im;
			}

			// We need to add bias for pos 0 only (see FFT output for (b, b, ..., b))
			output().real_[out_offset] += output().plane_size_ * filter.bias_.real();
			output().imag_[out_offset] += output().plane_size_ * filter.bias_.imag();
		}

		// for residuals
		for (int pos = 0; pos < input().length_; ++pos) {
			input_.dz_real_[pos] = 0;
			input_.dz_imag_[pos] = 0;
			input_.dz_star_real_[pos] = 0;
			input_.dz_star_imag_[pos] = 0;
		}
	}

	virtual void backward(int label) {
		backParams();

		// input
		FOREACH_PLANE(input_plane, input_offset, input_)
		{
			// d input / dz
			for (int in_pos = 0; in_pos < input_.plane_size_; ++in_pos) {
				std::complex<float> sum_dz = 0;
				std::complex<float> sum_dz_star = 0;

				FOREACH_PLANE(out_plane, out_offset, output())
				{
					FilterLayer& filter = filters_[out_plane];
					int j = out_offset + in_pos;
					std::complex<float> dLdzj(output().dz_real_[j], output().dz_imag_[j]);
					std::complex<float> dLdzj_star(output().dz_star_real_[j], output().dz_star_imag_[j]);

					std::complex<float> conv_pos(
							filter.data().real_[input_offset + in_pos],
							filter.data().imag_[input_offset + in_pos]);

					sum_dz += dLdzj * conv_pos;
					// sum_dz += dLdzj_star * (std::conj(dU_dz_star) == 0)

					// sum_dz_star += dLdzj_star * (dU_dz_star == 0);
					sum_dz_star += dLdzj_star * std::conj(conv_pos);
				}

				input_.dz_real_[input_offset + in_pos] += sum_dz.real();
				input_.dz_imag_[input_offset + in_pos] += sum_dz.imag();
				input_.dz_star_real_[input_offset + in_pos] += sum_dz_star.real();
				input_.dz_star_imag_[input_offset + in_pos] += sum_dz_star.imag();
			}
		}
	}

	void backParams() {
		// d filter / dz
		FOREACH_PLANE(out_plane, out_offset, output()) {  // for each filter
			FilterLayer& filter = filters_[out_plane];

			// bias - we need only dz_star_bias_
			std::complex<float> dLdzj_0(output().dz_real_[out_offset], output().dz_imag_[out_offset]);
			//filter.dz_bias_ += dLdzj_0;
			filter.dz_star_bias_ += dLdzj_0 * (float)input_.plane_size_;

			// params
			FOREACH_PLANE(in_plane, in_offset, filter.input())
			{
				for (int in_pos = 0; in_pos < filter.input().plane_size_; ++in_pos) {
					// std::complex<float> sum_dz = 0;
					std::complex<float> sum_dz_star = 0;

					for (int out_pos = 0; out_pos < filter.data().plane_size_; ++out_pos) {
						int j = out_offset + out_pos;
						std::complex<float> dLdzj(output().dz_real_[j], output().dz_imag_[j]);
						std::complex<float> dLdzj_star(output().dz_star_real_[j], output().dz_star_imag_[j]);

						std::complex<float> input_factor = u_root_.root(filter.index_->at(in_pos) * out_pos);

						// sum_dz += dLdzj * input_factor;
						// sum_dz += dLdzj_star * (std::conj(dU_dz_star) == 0)

						// sum_dz_star += dLdzj_star * (dU_dz_star == 0);
						sum_dz_star += dLdzj_star * std::conj(input_factor);
					}

					// filter.input_.dz_real_[in_offset + in_pos] = sum_dz.real();
					// filter.input_.dz_imag_[in_offset + in_pos] = sum_dz.imag();
					filter.input_.dz_star_real_[in_offset + in_pos] += sum_dz_star.real();
					filter.input_.dz_star_imag_[in_offset + in_pos] += sum_dz_star.imag();
				}
			}
		}
	}

	virtual void update(float learning_rate) {
		for (int f_indx = 0; f_indx < filters_.size(); ++f_indx) {
			filters_[f_indx].updateInput(learning_rate);
			filters_[f_indx].forward();
			filters_[f_indx].bias_ -= learning_rate * filters_[f_indx].dz_star_bias_;

//			filters_[f_indx].dz_bias_ = 0;
			filters_[f_indx].dz_star_bias_ = 0;

//			std::fill(filters_[f_indx].input_.dz_real_,
//					  filters_[f_indx].input_.dz_real_ + filters_[f_indx].input_.length_, 0.0);
//			std::fill(filters_[f_indx].input_.dz_imag_,
//					  filters_[f_indx].input_.dz_imag_ + filters_[f_indx].input_.length_, 0.0);
			std::fill(filters_[f_indx].input_.dz_star_real_,
					  filters_[f_indx].input_.dz_star_real_ + filters_[f_indx].input_.length_, 0.0);
			std::fill(filters_[f_indx].input_.dz_star_imag_,
					  filters_[f_indx].input_.dz_star_imag_ + filters_[f_indx].input_.length_, 0.0);
		}
	}

	int getNoFilters() {
		return filters_.size();
	}

	const FilterLayer* getFilter(int indx) const {
		assert(indx >= 0 && indx < filters_.size());
		return &filters_[indx];
	}

	void addFilter() {
		FilterLayer filter(&kernel_index_, this);
		filters_.push_back(filter);
	}

	void addFilter(FilterLayer& filter) {
		filters_.push_back(filter);
	}

	void setBias(int filter, std::complex<float> bias) {
		filters_[filter].setBias(bias);
	}

	const std::vector<int>* kernel_index() const {
		return &kernel_index_;
	}

	 FilterLayer* mutable_filter(int indx) {
		assert(indx >= 0 && indx < filters_.size());
		return &(filters_[indx]);
	}
};


#endif /* HADAMARD_H_ */
