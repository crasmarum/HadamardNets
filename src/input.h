/*
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
*/

#ifndef INPUT_H_
#define INPUT_H_

#include "data.h"

class InputLayer: public Layer {
	static FastFourierTransform<float> fft_;
public:
	InputLayer(int no_planes, int plane_size) : Layer(no_planes, plane_size) {
	}

	InputLayer(Layer *next) : Layer(next->input().no_planes_,
			next->input().plane_size_) {
		next_ = next;
	}

	virtual std::string getName() {
		return "InputLayer";
	}

	void getOutTemplate(int *no_planes, int *plane_size) {
		*no_planes = input_.no_planes_;
		*plane_size = input_.plane_size_;
	}

	virtual void forward() {
		FOREACH_PLANE(plane, offset, input_) {
			fft_.transform(input_.plane_size_, input_.real_ + offset, input_.imag_ + offset);
			std::copy(input().real_ + offset,
					  input().real_ + offset + input_.plane_size_, output().real_ + offset);
			std::copy(input().imag_ + offset,
					  input().imag_ + offset + input_.plane_size_, output().imag_ + offset);
		}
	}

	void forward(float *ext_real, float *ext_imag) {
		std::copy(ext_real, ext_real + input().length_, output().real_);
		std::copy(ext_imag, ext_imag + input().length_, output().imag_);
	}

	virtual void backward(int label) {
	}

	virtual void update(float learningRate) {
	}

	Data* mutable_input() {
		return &input_;
	}
};

#endif /* INPUT_H_ */
