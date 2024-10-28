/*
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
*/

#ifndef OUTPUT_H_
#define OUTPUT_H_

#include "data.h"
#include "activation.h"

class OutputLayer: public ULayer {
	Data output_;

public:
	OutputLayer(int plane_size) :
		ULayer(1, plane_size), output_(1, plane_size) {
		setNextLayer(this);
	}

	virtual std::string getName() {
		return "OutputLayer";
	}

	void getOutTemplate(int *no_planes, int *plane_size) {
	}

	virtual inline Data& output() {
		return output_;
	}

	float getProbability(int label) {
		return std::pow(output_.real_[label], 2) + std::pow(output_.imag_[label], 2);
	}

	float getLoss(int label) {
		assert(input_.no_planes_ == 1 && square_norm_.size() == 1 && label >= 0 && label < input_.plane_size_);
		float val = getProbability(label);
		val = (val < 1e-15 ? 1e-15: val);
		return - std::log(val);
	}

	virtual void backward(int label) {
		assert(input_.no_planes_ == 1 && square_norm_.size() == 1 && label >= 0 && label < input_.plane_size_);
		square_norm_[0] = (square_norm_[0] < 1e-15 ? 1e-15: square_norm_[0]);

		for (int pos = 0; pos < input_.plane_size_; ++pos) {
			if (label == pos) {
				float square_mod = std::pow(input_.real_[pos], 2) + std::pow(input_.imag_[pos], 2);
				square_mod = (square_mod < 1e-15 ? 1e-15: square_mod);
				input_.dz_star_real_[pos] = - input_.real_[pos]
										* (square_norm_[0] - square_mod) / (square_mod * square_norm_[0]);
				input_.dz_star_imag_[pos] = - input_.imag_[pos]
										* (square_norm_[0] - square_mod) / (square_mod * square_norm_[0]);
				input_.dz_real_[pos] = input_.dz_star_real_[pos];
				input_.dz_imag_[pos] = - input_.dz_star_imag_[pos];
			} else {
				input_.dz_star_real_[pos] = input_.real_[pos] / square_norm_[0];
				input_.dz_star_imag_[pos] = input_.imag_[pos] / square_norm_[0];
				input_.dz_real_[pos] = input_.dz_star_real_[pos];
				input_.dz_imag_[pos] = - input_.dz_star_imag_[pos];
			}
		}
	}

	Data& getInputForTest() {
		return input_;
	}

	int get_prediction() {
		float max = std::pow(output_.real_[0], 2) + std::pow(output_.imag_[0], 2);
		int ret = 0;
		for (int var = 1; var < output().length_; ++var) {
			float c_val = std::pow(output_.real_[var], 2) + std::pow(output_.imag_[var], 2);
			if (c_val > max) {
				ret = var;
				max = c_val;
			}
		}
		return ret;
	}
};

#endif /* OUTPUT_H_ */
