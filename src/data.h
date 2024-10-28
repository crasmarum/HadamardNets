/*
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
*/

#ifndef DATA_H_
#define DATA_H_

#define SWAP(a,b) temp=(a);(a)=(b);(b)=temp;

#include "assert.h"
#include <algorithm>
#include <complex>
#include <cmath>

#define FOREACH_PLANE(plane, offset, data) for (int offset = 0, plane = 0; offset < data.length_; offset += data.plane_size_, ++plane)
class Data {
private:
public:
	int no_planes_;
	int plane_size_;
	int length_;
	float* real_;
	float* imag_;
	float* dz_real_;
	float* dz_imag_;
	float* dz_star_real_;
	float* dz_star_imag_;

	Data() :
			no_planes_(0), plane_size_(0), length_(0), real_(0), imag_(0), dz_real_(
					0), dz_imag_(0), dz_star_real_(0), dz_star_imag_(0) {
	}
	Data(int no_planes, int plane_size) :
			no_planes_(no_planes), plane_size_(plane_size), length_(no_planes * plane_size) {
		assert(no_planes > 0 && plane_size > 0);
		real_ = new float[length_ * 6];
		imag_ = real_ + length_;
		dz_real_ = imag_ + length_;
		dz_imag_ = dz_real_ + length_;
		dz_star_real_ = dz_imag_ + length_;
		dz_star_imag_ = dz_star_real_ + length_;
		std::fill(real_, real_ + length_ * 6, 0.0);
	}

	static Data create_template(int no_planes, int plane_size) {
		Data ret;
		ret.no_planes_ = no_planes;
		ret.plane_size_ = plane_size;
		ret.length_ = no_planes * plane_size;
		return ret;
	}

	// Copy constructor.
	Data(const Data& other) : no_planes_(other.no_planes_), plane_size_(other.plane_size_),
			length_(other.length_), real_(0), imag_(0), dz_real_(0), dz_imag_(0),
			dz_star_real_(0), dz_star_imag_(0) {
		if (!other.real_) {
			return;
		}
		real_ = new float[length_ * 6];
		imag_ = real_ + length_;
		dz_real_ = imag_ + length_;
		dz_imag_ = dz_real_ + length_;
		dz_star_real_ = dz_imag_ + length_;
		dz_star_imag_ = dz_star_real_ + length_;
		std::copy(other.real_, other.real_ + length_ * 6, real_);
	}

	virtual ~Data() {
		if (real_) {
			delete[] real_;
			real_ = NULL;
		}
	}

	inline std::complex<float> z(int indx) const {
		assert(0 <= indx && indx < length_);
		return std::complex<float>(real_[indx], imag_[indx]);
	}

	inline std::complex<float> dz(int indx) const {
		assert(0 <= indx && indx < length_ && dz_real_ && dz_imag_);
		return std::complex<float>(dz_real_[indx], dz_imag_[indx]);
	}

	inline std::complex<float> dz_star(int indx) const {
		assert(0 <= indx && indx < length_ && dz_star_real_ && dz_star_imag_);
		return std::complex<float>(dz_star_real_[indx], dz_star_imag_[indx]);
	}
};

template<typename T>
class FastFourierTransform {
public:
	FastFourierTransform() {
	}

	static void transform(int length, T *real, T *imag) {
		if (length == 0) {
			return;
		}
		if ((length & (length - 1)) == 0) {  // power of 2
			transform_(length, real, imag);
			return;
		}

		transformBluestein(length, real, imag);
	}

	static void inverseTransform(int length, T *real, T *imag) {
		transform(length, imag, real);
	}

	static void convolve(int length, T *xreal, T *ximag, T *yreal, T *yimag) {
		if (length == 0) {
			return;
		}
		if ((length & (length - 1)) == 0) {
			convolve_(length, xreal, ximag, yreal, yimag);
			return;
		}

		int conv_len = 1;
		while (conv_len < length * 2 - 1) {
			conv_len *= 2;
		}

		T *areal = (T*) malloc(sizeof(T) * conv_len);
		T *aimag = (T*) malloc(sizeof(T) * conv_len);
		T *breal = (T*) malloc(sizeof(T) * conv_len);
		T *bimag = (T*) malloc(sizeof(T) * conv_len);
		for (int i = 0; i < length; i++) {
			areal[i] = xreal[i];
			aimag[i] = ximag[i];
			breal[i] = yreal[i];
			bimag[i] = yimag[i];
		}
		for (int i = length; i < conv_len; i++) {
			areal[i] = 0.0;
			aimag[i] = 0.0;
			breal[i] = 0.0;
			bimag[i] = 0.0;
		}

		convolve_(conv_len, areal, aimag, breal, bimag);

		for (int i = 0; i < length; i++) {
			yreal[i] = breal[i] + breal[length + i];
			yimag[i] = bimag[i] + bimag[length + i];
		}

		free(areal);
		free(aimag);
		free(breal);
		free(bimag);
	}

private:
	static inline unsigned int reverse(unsigned int val) {
		unsigned int ret = 0;
		unsigned int mask = 1U << 31;
		for (int var = 0; var < 32; ++var) {
			if (val & mask) {
				ret |= (1 << var);
			}
			mask = mask >> 1;
		}
		return ret;
	}

	static void convolve_(int length, T *xreal, T *ximag, T *yreal, T *yimag) {
		transform_(length, xreal, ximag);
		transform_(length, yreal, yimag);
		for (int i = 0; i < length; i++) {
			T temp = xreal[i] * yreal[i] - ximag[i] * yimag[i];
			ximag[i] = ximag[i] * yreal[i] + xreal[i] * yimag[i];
			xreal[i] = temp;
		}

		transform_(length, ximag, xreal);
		for (int i = 0; i < length; i++) {
			yreal[i] = xreal[i] / length;
			yimag[i] = ximag[i] / length;
		}
	}

	static void transform_(int length, T *real, T *imag) {
		if (length <= 1)
			return;

		int levels = -1;
		for (int i = 0; i < 32; i++) {
			if (1 << i == length) {
				levels = i;
				break;
			}
		}
		if (levels == -1)
			throw("Expected power of 2");

		T temp;
		for (int i = 0; i < length; i++) {
			int j = reverse(i) >> (32 - levels);
			if (j > i) {
				SWAP(real[i], real[j]);
				SWAP(imag[i], imag[j]);
			}
		}

		T *cosTable = (T*) malloc(sizeof(T) * length / 2);
		T *sinTable = (T*) malloc(sizeof(T) * length / 2);

		cosTable[0] = 1;
		sinTable[0] = 0;
		T qc = std::cos(2 * M_PI / length);
		T qs = std::sin(2 * M_PI / length);
		for (int i = 1; i < length / 2; i++) {
			cosTable[i] = cosTable[i - 1] * qc - sinTable[i - 1] * qs;
			sinTable[i] = sinTable[i - 1] * qc + cosTable[i - 1] * qs;
		}

		for (int size = 2; size <= length; size *= 2) {
			int halfsize = size / 2;
			int tablestep = length / size;
			for (int i = 0; i < length; i += size) {
				for (int j = i, k = 0; j < i + halfsize; j++, k += tablestep) {
					T tpre = real[j + halfsize] * cosTable[k]
							+ imag[j + halfsize] * sinTable[k];
					T tpim = -real[j + halfsize] * sinTable[k]
							+ imag[j + halfsize] * cosTable[k];
					real[j + halfsize] = real[j] - tpre;
					imag[j + halfsize] = imag[j] - tpim;
					real[j] += tpre;
					imag[j] += tpim;
				}
			}
		}

		free(sinTable);
		free(cosTable);
	}

	static void transformBluestein(int length, T *real, T *imag) {
		int conv_len = 1;
		while (conv_len < length * 2 - 1) {
			conv_len *= 2;
		}

		T *tc = (T*) malloc(sizeof(T) * 2 * length);
		T *ts = (T*) malloc(sizeof(T) * 2 * length);
		tc[0] = 1;
		ts[0] = 0;
		T qc = std::cos(M_PI / length);
		T qs = std::sin(M_PI / length);
		for (int i = 1; i < 2 * length; i++) {
			tc[i] = tc[i - 1] * qc - ts[i - 1] * qs;
			ts[i] = ts[i - 1] * qc + tc[i - 1] * qs;
		}

		T *cosTable = (T*) malloc(sizeof(T) * length);
		T *sinTable = (T*) malloc(sizeof(T) * length);
		for (int i = 0; i < length; i++) {
			int j = (int) (((long) i * i) % (length * 2));
			cosTable[i] = tc[j];
			sinTable[i] = ts[j];
		}

		T *areal = (T*) malloc(sizeof(T) * conv_len);
		T *aimag = (T*) malloc(sizeof(T) * conv_len);
		T *breal = (T*) malloc(sizeof(T) * conv_len);
		T *bimag = (T*) malloc(sizeof(T) * conv_len);
		for (int i = length; i < conv_len; i++) {
			areal[i] = 0.0;
			aimag[i] = 0.0;
			breal[i] = 0.0;
			bimag[i] = 0.0;
		}

		for (int i = 0; i < length; i++) {
			areal[i] = real[i] * cosTable[i] + imag[i] * sinTable[i];
			aimag[i] = -real[i] * sinTable[i] + imag[i] * cosTable[i];
		}

		breal[0] = cosTable[0];
		bimag[0] = sinTable[0];
		for (int i = 1; i < length; i++) {
			breal[i] = breal[conv_len - i] = cosTable[i];
			bimag[i] = bimag[conv_len - i] = sinTable[i];
		}

		convolve_(conv_len, areal, aimag, breal, bimag);

		for (int i = 0; i < length; i++) {
			real[i] = breal[i] * cosTable[i] + bimag[i] * sinTable[i];
			imag[i] = -breal[i] * sinTable[i] + bimag[i] * cosTable[i];
		}

		free(tc);
		free(ts);
		free(sinTable);
		free(cosTable);
		free(areal);
		free(aimag);
		free(breal);
		free(bimag);
	}
};

#endif /* DATA_H_ */
