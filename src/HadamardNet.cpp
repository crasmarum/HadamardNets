/*
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
*/

#include <iostream>

#include "activation.h"
#include "data.h"
#include "hadamard.h"
#include "input.h"
#include "layer.h"
#include "output.h"
#include "net.h"
#include "unityroots.h"

std::string mnist_test_images = "data/t10k-images.idx3-ubyte";
std::string mnist_test_labels = "data/t10k-labels.idx1-ubyte";
std::string mnist_fft_data = "data/fft_10k_test.fft";
std::string mnist_saved_model = "model/94percent_mnist_h7x7.mod";

// This code:
// 1. Creates test data by applying FFT<28*28> to the entries from the
//    MNIST test corpus.
// 2. Loads an already saved model from the model/94percent_mnist_h7x7.mod
//    file.
// 3. Computes the accuracy of the model against the test data.
//
int main() {
	std::cout << "Creating test data from " << mnist_test_images << "..." << std::endl;
	Mnist mnist_utils;
	mnist_utils.createData(mnist_test_images, mnist_test_labels, 10000, 5, mnist_fft_data);

	CpuNet net;
	std::cout << std::endl << "Loading model " << mnist_saved_model << "..."  << std::endl;
	assert(net.restore(mnist_saved_model));
	std::cout << net.toString() << std::endl;

	FftDataReader reader;
	assert(reader.Open_Fft(mnist_fft_data, 10000));
	assert(net.layer_at(0)->input().length_ == reader.data_length());

	std::cout << std::endl << "Computing model accuracy..."  << std::endl;
	float *real, *imag;
	int *label;
	float predicted = 0;
	int total = 0;
	while (reader.ReadBatch(1, &real, &imag, &label)) {
		net.cpuForward(real, imag);
		OutputLayer *layer = static_cast<OutputLayer*>(net.cpu_back());
		if (layer->get_prediction() == *label) {
			predicted++;
		}
		total++;
		if (total % 1000 == 0) {
			printf("Current accuracy for %d entries: %f\n", total, predicted / total);
		}
	}
	assert(total);
	printf("Final accuracy: %f = %f / %d\n", predicted / total, predicted, total);

	std::cout << "Done." << std::endl;
	return 0;
}
