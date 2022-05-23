#include <cmath>
#include <iostream>
#include <ostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <complex>

const double PI = 3.14159265359;
const double PI_2 = 2 * PI;

template <typename Type>
void PrintVector(const std::vector<Type>& S) {
	// Печать вектора в консоль
	for (const Type& num : S) {
		std::cout << num << "  ";
	}
	std::cout << '\n';
}

template <typename Type>
void WriteToTxt(const std::vector<Type>& arr, const std::string& name) {
	// Запись в файл вектора
	std::ofstream file;
	file.open(name);
	for (const Type& num : arr) {
			file << num << std::endl;
	}
	file.close();
}

template <typename Type> 
void Print2dVector(const std::vector<std::vector<Type>>& vec) {
	for (const std::vector<Type>& nums : vec) {
		for (const Type& num : nums) {
			std::cout << num << "\t";
		}
		std::cout << std::endl;
	} 
	std::cout << '\n';
}

void Print2dVectorComplex(const std::vector<std::vector<std::complex<double>>>& vec) {
	for (const std::vector<std::complex<double>>& nums : vec) {
		for (const std::complex<double>& num : nums) {
			std::cout << num.real() << " " << num.imag() << std::endl;
		}			
	}
}

template <typename Type>
void VectorTo2dVector(std::vector<std::vector<Type>>& vec2d, const std::vector<Type>& vec) {
		for (size_t i = 0; i < vec2d.size(); ++i) {
				for (size_t j = 0; j < vec2d[0].size(); ++j) {
						vec2d[i][j] = vec[i * vec2d[0].size() + j];
				}
		}
}

void RandBits(std::vector<uint32_t>& bits) {
	// Генерация битовой последовательности
	std::random_device rnd;
	std::mt19937 gen(rnd());
	std::uniform_int_distribution<uint32_t> number(0, 1);
	for (uint32_t& bit : bits) {
		bit = number(gen);
	}
}

std::vector<std::vector<uint32_t>> Coding(const std::vector<std::vector<uint32_t>>& out_bits) {
	std::vector<std::vector<uint32_t>> code_sequences (out_bits.size(), std::vector<uint32_t> (8));
	for (size_t i = 0; i < out_bits.size(); ++i) {
		std::copy(out_bits[i].begin(), out_bits[i].end(), code_sequences[i].begin() + 1);
		code_sequences[i][5] = (out_bits[i][0] + out_bits[i][1] + out_bits[i][2]) % 2;
		code_sequences[i][6] = (out_bits[i][1] + out_bits[i][2] + out_bits[i][3]) % 2;
		code_sequences[i][7] = (out_bits[i][0] + out_bits[i][1] + out_bits[i][3]) % 2;
	}
	return code_sequences;
}

std::vector<std::vector<std::complex<double>>> Modulation (const std::vector<std::vector<uint32_t>> code_sequences, double& A) {
	std::vector<std::vector<std::complex<double>>> signal (code_sequences.size(), std::vector<std::complex<double>> (8));
	for (size_t i = 0; i < signal.size(); ++i) {
		for (size_t j = 0; j < signal[0].size(); ++j) {
			if (code_sequences[i][j]) {
				A = -A;
				signal[i][j] = std::complex<double>(A, 0);
			} else {
				signal[i][j] = std::complex<double>(A, 0);
			}
		}
	}
	return signal;
}


std::vector<std::vector<std::complex<double>>> AddNormalNoise (const std::vector<std::vector<std::complex<double>>> signal) {
	std::vector<std::vector<std::complex<double>>> signal_with_noise (signal.size(), std::vector<std::complex<double>> (signal[0].size()));
	double mean = 0;
	std::mt19937 generator(rand());
	double standart_deviation = 1;
	std::normal_distribution<double> distribution(mean,standart_deviation);
	for (size_t i = 0; i < signal_with_noise.size(); ++i) {
		for (size_t j = 0; j < signal_with_noise[0].size(); ++j) {
				signal_with_noise[i][j] = signal[i][j] + std::complex(distribution(generator), distribution(generator));
			}		
		}
	return signal_with_noise;
}


std::vector<std::vector<uint32_t>> Demodulation (const std::vector<std::vector<std::complex<double>>>& signal_with_noise) {
	std::vector<std::vector<uint32_t>> in_bits (signal_with_noise.size(), std::vector<uint32_t> (signal_with_noise[0].size() - 1));
	for (size_t i = 0; i < signal_with_noise.size(); ++i) {
		std::complex<double> signal_complex_0_T = signal_with_noise[i][0];
		for (size_t j = 0; j < signal_with_noise[0].size() - 1; ++j) {
			std::complex<double> signal_complex_T_2T = signal_with_noise[i][j + 1];
			double phi = std::arg(signal_complex_0_T / signal_complex_T_2T);
			if (phi > -1.57079632679 && phi < 1.57079632679) {
					in_bits[i][j] = 0;
			} else {
					in_bits[i][j] = 1;
			}
			signal_complex_0_T = signal_complex_T_2T;
		}		
	}
	return in_bits;
}

uint32_t ModTwoAddVectors(const std::vector<uint32_t>& vec1, const std::vector<uint32_t> vec2) {
	uint32_t result = 0;
	for (size_t i = 0; i < vec1.size(); ++i) {
			result += (vec1[i + 1] + vec2[i]) % 2;
	}
	return result;
}

double CheckError(const std::vector<std::vector<uint32_t>>& out_bits, const std::vector<std::vector<uint32_t>>& in_bits) {
	double errs = 0;
	for (size_t i = 0; i < in_bits.size(); ++i) {
		errs += ModTwoAddVectors(out_bits[i], in_bits[i]);
	}
	return errs / static_cast<double>(in_bits.size());
}

int main () {
	uint32_t block = 1000000;
	std::vector<std::vector<uint32_t>> out_bits (block, std::vector<uint32_t> (4, 0));
	std::vector<std::vector<uint32_t>> out_code_sequences = Coding(out_bits);
	std::vector<uint32_t> h = {0, 1, 2, 3};
	std::vector<double> errs (4);
	double A = 2.43 * sqrt(2);
	std::vector<std::vector<std::complex<double>>> signal =  Modulation(out_code_sequences, A);
	std::vector<std::vector<std::complex<double>>> signal_with_noise = AddNormalNoise(signal);
	std::vector<std::vector<uint32_t>> in_code_sequences = Demodulation(signal_with_noise);
	std::cout << CheckError(out_code_sequences, in_code_sequences) << std::endl;
	return 0;
}
