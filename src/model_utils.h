/*
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
*/

#ifndef CPU_MODEL_SAVER_H_
#define CPU_MODEL_SAVER_H_

#include <stdlib.h>
#include <cstring>
#include <fstream>
#include <ios>
#include <vector>

#include "layer.h"
#include "fullyconnected.h"
#include "hadamard.h"
#include "input.h"
#include "output.h"


const int __one__ = 1;
const bool isCpuLittleEndian = 1 == *(char*) (&__one__);

const int magicNo = 0xABCDE0EE;
const int isULayer = 0;
const int isHadamard = 1;
const int isFullyConn = 2;
const int isOutput = 3;
const int isInput = 4;
const int isDiv = 5;

class BinaryReader {
	std::ifstream infile_;
public:

	bool read_int32(int32_t* read) {
		char buffer[4];
		if (!infile_.read(buffer, 4)) {
			return false;
		}
		char *pInt = (char*) read;
		if (isCpuLittleEndian) {
			for (int i = 0; i < 4; ++i) {
				pInt[3 - i] = buffer[i];
			}
		} else {
			for (int i = 0; i < 4; ++i) {
				pInt[i] = buffer[i];
			}
		}

		return true;
	}

	static bool read_int32(std::ifstream& stream, int32_t* read) {
		char buffer[4];
		if (!stream.read(buffer, 4)) {
			return false;
		}
		char *pInt = (char*) read;
		if (isCpuLittleEndian) {
			for (int i = 0; i < 4; ++i) {
				pInt[3 - i] = buffer[i];
			}
		} else {
			for (int i = 0; i < 4; ++i) {
				pInt[i] = buffer[i];
			}
		}

		return true;
	}

	bool read_double(double* read) {
		char buffer[8];
		if (!infile_.read(buffer, 8)) {
			return false;
		}

		char *pDouble = (char*) read;
		if (isCpuLittleEndian) {
			for (int i = 0; i < 8; ++i) {
				pDouble[7 - i] = buffer[i];
			}
		} else {
			for (int i = 0; i < 8; ++i) {
				pDouble[i] = buffer[i];
			}
		}

		return true;
	}

	bool read_float(float* floa, int count) {
		for (int var = 0; var < count; ++var) {
			if (!read_float(floa + var))
				return false;
		}
		return true;
	}

	bool read_float(float* floa) {
		char buffer[4];
		if (!infile_.read(buffer, 4)) {
			return false;
		}
		char *pFloat = (char*) floa;
		if (isCpuLittleEndian) {
			for (int i = 0; i < 4; ++i) {
				pFloat[3 - i] = buffer[i];
			}
		} else {
			for (int i = 0; i < 4; ++i) {
				pFloat[i] = buffer[i];
			}
		}

		return true;
	}

	static bool read_float(std::ifstream& stream, float* floa) {
		char buffer[4];
		if (!stream.read(buffer, 4)) {
			return false;
		}
		char *pFloat = (char*) floa;
		if (isCpuLittleEndian) {
			for (int i = 0; i < 4; ++i) {
				pFloat[3 - i] = buffer[i];
			}
		} else {
			for (int i = 0; i < 4; ++i) {
				pFloat[i] = buffer[i];
			}
		}

		return true;
	}

	bool read_chars(char* data, int length) {
		return (bool) infile_.read(data, length);
	}

	bool read_unsigned_chars(unsigned char* data, int length) {
		return (bool) infile_.read((char*) data, length);
	}

	bool open(std::string file) {
		infile_.open(file, std::ofstream::in | std::ofstream::binary);
		if (infile_) {
			return true;
		}
		return false;
	}

	void close() {
		infile_.close();
	}
};

class ModelSaver {
	bool ReadDataTemplate(BinaryReader& reader, Data *data) {
		if (!reader.read_int32(&data->no_planes_))
			return false;
		if (!reader.read_int32(&data->plane_size_))
			return false;
		data->length_ = data->no_planes_ * data->plane_size_;

		return true;
	}

	bool RestoreUlayer(std::vector<Layer*>& net, BinaryReader& reader) {
		int isResidual;
		if (!reader.read_int32(&isResidual))
			return false;

		Data data;
		if (!ReadDataTemplate(reader, &data))
			return false;

		ULayer *layer = new ULayer(data.no_planes_, data.plane_size_);

		if (net.size()) {
			// last layer's output is next layer' input
			net.back()->setNextLayer(layer);
		}
		net.push_back(layer);

		return true;
	}

	bool RestoreHadamard(std::vector<Layer*>& net, BinaryReader& reader) {
		Data data;
		if (!ReadDataTemplate(reader, &data))
			return false;
		if (!data.length_)
			return false;

		int size, tmp;

		reader.read_int32(&size);
		assert(size);
		std::vector<int> kernel_index;
		for (int var = 0; var < size; ++var) {
			reader.read_int32(&tmp);
			kernel_index.push_back(tmp);
		}
		HadamardLayer *hadam = new HadamardLayer(data.no_planes_,
				data.plane_size_, kernel_index);

		reader.read_int32(&size);
		assert(size);
		float real, imag;
		for (int var = 0; var < size; ++var) {
			hadam->addFilter();
			FilterLayer *filter = hadam->mutable_filter(var);

			reader.read_float(&real);
			reader.read_float(&imag);
			filter->setBias(std::complex<float>(real, imag));

			if (!ReadDataTemplate(reader, &data))
				return false;
			if (data.no_planes_ != filter->input().no_planes_)
				return false;
			if (data.plane_size_ != filter->input().plane_size_)
				return false;

			reader.read_float(filter->mutable_input()->real_,
					filter->mutable_input()->length_);
			reader.read_float(filter->mutable_input()->imag_,
					filter->mutable_input()->length_);
			filter->forward();
		}

		if (net.size()) {
			// last layer's output is next layer's input
			net.back()->setNextLayer(hadam);
		}
		net.push_back(hadam);

		return true;
	}

	bool RestoreFullyConn(std::vector<Layer*>& net, BinaryReader& reader) {
		Data data;
		if (!ReadDataTemplate(reader, &data))
			return false;
		if (!data.length_)
			return false;
		int out_length;
		if (!reader.read_int32(&out_length) || !out_length)
			return false;

		FullyConn *f_conn = new FullyConn(data.no_planes_, data.plane_size_,
				out_length);
		float real, imag;
		for (int f_indx = 0; f_indx < out_length; ++f_indx) {
			if (!reader.read_float(&real))
				return false;
			if (!reader.read_float(&imag))
				return false;
			*(f_conn->mutable_bias(f_indx)) = std::complex<float>(real, imag);

			reader.read_float(f_conn->mutable_filter(f_indx)->real_,
					f_conn->mutable_filter(f_indx)->length_);
			reader.read_float(f_conn->mutable_filter(f_indx)->imag_,
					f_conn->mutable_filter(f_indx)->length_);
		}

		if (net.size()) {
			// last layer's output is next layer's input
			net.back()->setNextLayer(f_conn);
		}
		net.push_back(f_conn);

		return true;
	}

	bool RestoreInputLayer(std::vector<Layer*>& net, BinaryReader& reader) {
		Data data;
		if (!ReadDataTemplate(reader, &data))
			return false;
		if (!data.length_)
			return false;

		InputLayer* input = new InputLayer(data.no_planes_, data.plane_size_);

		if (net.size()) {
			// last layer's output is next layer's input
			net.back()->setNextLayer(input);
		}
		net.push_back(input);
		return true;
	}

	bool RestoreOutputLayer(std::vector<Layer*>& net, BinaryReader& reader) {
		Data data;
		if (!ReadDataTemplate(reader, &data))
			return false;

		OutputLayer *layer = new OutputLayer(data.plane_size_);
		if (net.size()) {
			// last layer's output is next layer' input
			net.back()->setNextLayer(layer);
		}
		net.push_back(layer);

		return true;
	}

public:

	bool Restore(std::vector<Layer*>& net, std::string file) {
		BinaryReader reader;
		if (!reader.open(file)) {
			//std::cout << "cannot open: " << file << std::endl;
			return false;
		}

		int current;
		if (!reader.read_int32(&current) || current != magicNo) {
			//std::cout << std::hex << current << std::endl;
			reader.close();
			return false;
		}
		while (reader.read_int32(&current)) {
			if (current == isInput && RestoreInputLayer(net, reader)) {
				continue;
			}
			if (current == isHadamard && RestoreHadamard(net, reader)) {
				continue;
			}
			if (current == isFullyConn && RestoreFullyConn(net, reader)) {
				continue;
			}
			if (current == isOutput && RestoreOutputLayer(net, reader)) {
				continue;
			}
			if (current == isULayer && RestoreUlayer(net, reader)) {
				continue;
			}

			reader.close();
			return false;
		}

		reader.close();
		return true;
	}

	virtual ~ModelSaver() {
	}
	;
};

class DataReader {
	friend class FftDataReader;
	int plane_size_;
	int no_planes_;
public:
	static const int kMagicNo = 0xABCDE0EE;
	DataReader() :
			plane_size_(0), no_planes_(0) {
	}
	virtual ~DataReader() {
	}
	;
	virtual int ReadBatch(int batch_size, float **real, float **imag,
			int **label) = 0;
	virtual bool Open_Fft(std::string image_file, int count) {
		return false;
	}
	void getTemplate(int *planes, int *plane_size) {
		*planes = no_planes_;
		*plane_size = plane_size_;
	}
};

class FftDataReader: public DataReader {
	std::ifstream images_;
	float *real_;
	float *imag_;
	int *label_;

	int current_indx_;
	int max_count_;
	int data_length_;

public:
	FftDataReader() :
			real_(NULL), imag_(NULL), label_(NULL), current_indx_(0), max_count_(
					0), data_length_(0) {
	}
	~FftDataReader() {
		if (label_)
			delete (label_);
		if (imag_)
			delete (imag_);
		if (real_)
			delete (real_);
	}

	virtual bool Open_Fft(std::string image_file, int no_entries) {
		std::cout << "reading data from " << image_file << "..." << std::endl;
		images_.open(image_file.c_str(), std::ios::in | std::ios::binary);
		if (images_.fail())
			return false;

		assert(BinaryReader::read_int32(images_, &no_planes_));
		assert(no_planes_ == kMagicNo);
		assert(BinaryReader::read_int32(images_, &no_planes_));
		assert(BinaryReader::read_int32(images_, &plane_size_));
		data_length_ = no_planes_ * plane_size_;
		std::cout << "planes: " << no_planes_ << " plane_size: " << plane_size_
				<< std::endl;

		max_count_ = no_entries;
		real_ = (float*) malloc(data_length_ * max_count_ * sizeof(float));
		imag_ = (float*) malloc(data_length_ * max_count_ * sizeof(float));
		label_ = (int*) malloc(max_count_ * sizeof(int));

		int current = 0;
		for (int entry = 0; entry < max_count_; ++entry) {
			for (int var = 0; var < data_length_; ++var) {
				assert(
						BinaryReader::read_float(images_,
								real_ + data_length_ * entry + var));
				assert(
						BinaryReader::read_float(images_,
								imag_ + data_length_ * entry + var));
			}
			assert(BinaryReader::read_int32(images_, label_ + entry));
			current++;
		}

		images_.close();
		std::cout << "read: " << max_count_ << " entries." << std::endl;
		return data_length_;
	}

	virtual void Reset() {
		current_indx_ = 0;
	}

	virtual int ReadBatch(int batch_size, float **real, float **imag,
			int **label) {
		if (current_indx_ + batch_size > max_count_) {
			return 0;
		}
		*real = real_ + current_indx_ * data_length_;
		*imag = imag_ + current_indx_ * data_length_;
		*label = label_ + current_indx_;
		current_indx_ += batch_size;
		return batch_size;
	}

	int data_length() {
		return data_length_;
	}
};

class BinaryWriter {
	std::string file_;
	std::ofstream outfile_;

public:
	bool open(std::string file) {
		file_ = file;
		outfile_.open(file, std::ofstream::out | std::ofstream::trunc
				| std::ofstream::binary);
		if (outfile_) {
			return true;
		}
		return false;
	}

	void close() {
		outfile_.close();
	}

	void write(double* pDouble, int size) {
		for (int var = 0; var < size; ++var) {
			write_double(*(pDouble + var));
		}
	}

	void write(float* pFloat, int size) {
		for (int var = 0; var < size; ++var) {
			write_float(*(pFloat + var));
		}
	}

	void write_double(double doubl) {
		if (isCpuLittleEndian) {
			char data[8], *pDouble = (char*) (double*) (&doubl);
			for (int i = 0; i < 8; ++i) {
				data[i] = pDouble[7 - i];
			}
			outfile_.write(data, 8);
		} else {
			outfile_.write((char*) (&doubl), 8);
		}
	}

	void write_float(float floa) {
		if (isCpuLittleEndian) {
			char data[4], *pFloat = (char*) (float*) (&floa);
			for (int i = 0; i < 4; ++i) {
				data[i] = pFloat[3 - i];
			}
			outfile_.write(data, 4);
		} else {
			outfile_.write((char*) (&floa), 4);
		}
	}

	void write_int32(int32_t v) {
		if (isCpuLittleEndian) {
			char data[8], *pDouble = (char*) (double*) (&v);
			for (int i = 0; i < 4; ++i) {
				data[i] = pDouble[3 - i];
			}
			outfile_.write(data, 4);
		} else {
			outfile_.write((char*) (&v), 4);
		}
	}

	void write_chars(const char *data, int length) {
		outfile_.write(data, length);
	}

	void write_unsigned_chars(const unsigned char *data, int length) {
		outfile_.write((char*) data, length);
	}

	std::string file() {
		return file_;
	}
};



class MnistDataReader {
	std::ifstream images_;
	std::ifstream labels_;
	bool is_open_;
	char *local_data_;
	char *local_labels_;
	int current_indx_;
	int max_count_;
	FastFourierTransform<float> fft_;

	bool open(std::string image_file, std::string label_file) {
		assert(!is_open_);
	    images_.open(image_file.c_str(), std::ios::in | std::ios::binary);
	    if (images_.fail()) return false;
	    labels_.open(label_file.c_str(), std::ios::in | std::ios::binary );
	    if (labels_.fail()) return false;

		// Reading file headers
	    char number;
	    for (int i = 1; i <= 16; ++i) {
	        images_.read(&number, sizeof(char));
		}
	    for (int i = 1; i <= 8; ++i) {
	    	labels_.read(&number, sizeof(char));
	    }
	    is_open_ = true;
		return true;
	}

public:

	bool Read(Data *data, int *label) {
		assert(is_open_);
		assert(data->length_ == 28 * 28);

		if (current_indx_ >= max_count_) return false;
		*label = (int) local_labels_[current_indx_];
		for (int indx = 0; indx <  28 * 28; ++indx) {
			data->real_[indx] = (local_data_[current_indx_ * 28 * 28 + indx] == 0 ? 0.0f : 1.0f);
			data->imag_[indx] = 0.0;
		}
		current_indx_++;
		return true;
	}

	bool Open(std::string image_file, std::string label_file, int count) {
		if (!open(image_file, label_file)) return false;
		is_open_ = false;

		local_data_ = (char*) malloc(count * 28 * 28 * sizeof(char));
		if (!local_data_) return false;
		if (!images_.read(local_data_, count * 28 * 28 * sizeof(char))) return false;

		local_labels_ = (char*) malloc(count * sizeof(char));
		if (!local_labels_) return false;
		if (!labels_.read(local_labels_, count * sizeof(char))) return false;

		max_count_ = count;
		std::cout << "read: " << max_count_ << std::endl;
		is_open_ = true;
		return true;
	}

	MnistDataReader() : is_open_(false), local_data_(NULL),
			local_labels_(NULL), current_indx_(0), max_count_(0) {}

	~MnistDataReader() {
		if (local_data_) delete(local_data_);
		if (local_labels_) delete(local_labels_);
		images_.close();
		labels_.close();
	}
};


class Mnist {
public:
	Mnist() {}

	static void createData(std::string mnist_in_images, std::string mnist_in_labels,
			               int mnist_entries, int mnist_zero_freq, std::string mnist_out_file) {
		BinaryWriter writer;
		assert(writer.open(mnist_out_file));

		MnistDataReader reader;
		assert(reader.Open(mnist_in_images, mnist_in_labels, mnist_entries));

		Data input(1, 28 * 28);
		int indx = 0;

		writer.write_int32(DataReader::kMagicNo);
		writer.write_int32(1);
		writer.write_int32(28 * 28);

		printf("Writing FFT(input) to the %s file.\n", mnist_out_file.c_str());
		if (mnist_zero_freq > 0) {
			printf("\tNote: will zero the first %d frequencies.\n", mnist_zero_freq);
		}

		FastFourierTransform<float> fft;
		for (int var = 0; var < mnist_entries; ++var) {
			int label = 99999;
			assert(reader.Read(&input, &label));
			assert(label < 10 && label >= 0);
			fft.transform(input.length_, input.real_, input.imag_);
			indx++;

			for (int var = 0; var < input.length_; ++var) {
				if (var < mnist_zero_freq) {
					writer.write_float(0);
					writer.write_float(0);
					continue;
				}
				writer.write_float(input.real_[var]);
				writer.write_float(input.imag_[var]);
			}
			writer.write_int32(label);
			if (indx % 1000 == 0) {
				printf("wrote %d entries.\n", indx);
			}
		}
		writer.close();
		assert(indx == mnist_entries);
	}
};

#endif /* CPU_MODEL_SAVER_H_ */
