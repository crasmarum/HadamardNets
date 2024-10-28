/*
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
*/

#ifndef ACTIVATION_H_
#define ACTIVATION_H_

#include "assert.h"
#include <algorithm>
#include <cfloat>
#include <vector>

#include "data.h"
#include "layer.h"

class ULayer: public Layer {
	friend class OutputLayer;

	std::vector<float> square_norm_;
public:

	ULayer(int no_planes, int plane_size) :
			Layer(no_planes, plane_size), square_norm_(no_planes) {
		std::fill (square_norm_.begin(), square_norm_.begin() + no_planes, 0.0);
	}

	virtual ~ULayer() {
	}

	virtual std::string getName() {
		return "ULayer";
	}

	void getOutTemplate(int *no_planes, int *plane_size) {
		*no_planes = input_.no_planes_;
		*plane_size = input_.plane_size_;
	}

	virtual void forward() {
		assert(
				next_ && next_->input_.no_planes_ == input_.no_planes_
						&& next_->input_.plane_size_ == input_.plane_size_);

		FOREACH_PLANE(plane, offset, output())
		{
			double sum = 0.0;
			for (int pos = 0; pos < input_.plane_size_; ++pos) {
				sum += (pow(input_.real_[offset + pos], 2)
						+ pow(input_.imag_[offset + pos], 2));
			}

			square_norm_[plane] = sum;
			sum = sqrt(sum);

			assert(sum > FLT_EPSILON);
			for (int pos = 0; pos < input_.plane_size_; ++pos) {
				output().real_[offset + pos] = (float) input_.real_[offset
						+ pos] / sum;
				output().imag_[offset + pos] = (float) input_.imag_[offset
						+ pos] / sum;
			}
		}
	}

	// U_j : C^|input| -> C
	// k = 1..|input|, j = 1..|output|
	inline std::complex<float> dU_dz(bool k_eq_j, std::complex<float> zk, std::complex<float> zj,
									 float in_plane_sq_norm_, float in_plane_sq_norm_3_2) {
		if (k_eq_j) {
			// 1\|Z| - 1/2 |z_k|^2 / |Z|^3
			float abs_zk_sq = std::pow(zk.real(), 2) + std::pow(zk.imag(), 2);
			abs_zk_sq = (abs_zk_sq < 1e-15 ? 1e-15: abs_zk_sq);

			return std::complex<float>(0.5 * (2 * in_plane_sq_norm_- abs_zk_sq) / in_plane_sq_norm_3_2, 0.0);
		}

		// -1/2 z_j z_k* / |Z|^3 for j!= k
		return -0.5f * zj * std::conj<float>(zk) / in_plane_sq_norm_3_2;
	}

	// U_j : C^|input| -> C
	// k = 1..|input|, j = 1..|output|
	inline std::complex<float> dU_dz_star(bool k_eq_j, std::complex<float> zk, std::complex<float> zj,
			                              float in_plane_sq_norm_3_2) {
		if (k_eq_j) {
			// -1/2 z_k^2 / |Z|^3
			return -0.5f * zk * zk / in_plane_sq_norm_3_2;
		}

		// -1/2 z_j z_k \ |Z|^3 for j!= k
		return  -0.5f * zj * zk / in_plane_sq_norm_3_2;
	}

	// U : C^|input| -> C^|output|, U = (U1, ... U_j, ... U_|output()|), U_j : C^|input| -> C,
	// L : C^|output()| -> R
	// k = 1..|input|, j = 1..|output|
	// (d L(U) / dz_k) = sum_j (dL(U)/dz_j) (dU_j / dz_k) + sum_j (dL(U)/dz_j*) [(dU_j*/dz_k) == (dU_j / dz_k*)*]
	// (d L(U) / dz_k*) = sum_j (dL(U)/dz_j) (dU_j / dz_k*) + sum_j (dL(U)/dz_j*) [(dU_j*/dz_k*) == (dU_j / dz_k)*]
	virtual void backward(int label) {
		assert(next_ && input_.no_planes_ == output().no_planes_ && input_.plane_size_ == output().plane_size_
				&& square_norm_.size() == input_.no_planes_);
		FOREACH_PLANE(input_plane, input_offset, input_) {
			float in_plane_sq_norm_ = (square_norm_[input_plane] < 1e-15 ? 1e-15: square_norm_[input_plane]);
			float in_plane_sq_norm_3_2 = std::pow(in_plane_sq_norm_, 1.5);

			for (int in_pos = 0; in_pos < input_.plane_size_; ++in_pos) {
				std::complex<float> sum_dz = 0;
				std::complex<float> sum_dz_star = 0;
				std::complex<float> zk(input_.real_[input_offset + in_pos],
						               input_.imag_[input_offset + in_pos]);

				for (int out_pos = 0; out_pos < output().plane_size_; ++out_pos) {
					int j = input_offset + out_pos;
					std::complex<float> zj(input_.real_[j], input_.imag_[j]);
					std::complex<float> dLdzj(output().dz_real_[j], output().dz_imag_[j]);
					std::complex<float> dLdzj_star(output().dz_star_real_[j], output().dz_star_imag_[j]);

					sum_dz += dLdzj * dU_dz(in_pos == out_pos, zk, zj, in_plane_sq_norm_, in_plane_sq_norm_3_2);
					sum_dz += dLdzj_star * std::conj(dU_dz_star(in_pos == out_pos, zk, zj, in_plane_sq_norm_3_2));

					sum_dz_star += dLdzj * dU_dz_star(in_pos == out_pos, zk, zj, in_plane_sq_norm_3_2);
					sum_dz_star += dLdzj_star * std::conj(
							dU_dz(in_pos == out_pos, zk, zj, in_plane_sq_norm_, in_plane_sq_norm_3_2));

				}

				input_.dz_real_[input_offset + in_pos] = sum_dz.real();
				input_.dz_imag_[input_offset + in_pos] = sum_dz.imag();
				input_.dz_star_real_[input_offset + in_pos] = sum_dz_star.real();
				input_.dz_star_imag_[input_offset + in_pos] = sum_dz_star.imag();
			}
		}
	}

	float square_norm(int plane) {
		assert(plane >= 0 && plane < input_.no_planes_);
		return square_norm_[plane];
	}

	virtual void update(float learningRate) {
	}
};

#endif /* ACTIVATION_H_ */
