#include <cmath>
#include <iostream>
#include <ostream>
#include <vector>
#include <random>
#include <map>
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

template <typename Type>
void VectorTo2dVector(std::vector<std::vector<Type>>& vec2d, const std::vector<Type>& vec) {
		for (size_t i = 0; i < vec2d.size(); ++i) {
				for (size_t j = 0; j < vec2d[0].size(); ++j) {
						vec2d[i][j] = vec[i * vec2d[0].size() + j];
				}
		}
}


std::vector<double> Modulation(std::vector<uint32_t>& bits,const double& A, const double& Fn, const double& Fd) {
		uint32_t period = Fd / Fn;
		double t = 1. / Fd;
		double phase_shift = 0;
		auto it = bits.begin();
		it = bits.insert(it, 0);
		std::vector<double> signal (period * bits.size());
		for (size_t i = 0; i < bits.size(); ++i) {
					if (bits[i] == 0) {
						for (size_t j  = 0 ; j < period; ++j) {
									signal[i * period + j] = A * cos(PI_2 * j * t * Fn + phase_shift);
							}
						} else {
							phase_shift += PI;
							for (size_t j  = 0 ; j < period; ++j) {
									signal[i * period + j] = A * cos(PI_2 * j * t * Fn + phase_shift);
							}
						}				
				}
		return signal;
}

std::vector<double> SignalModel(const double& A, const double& Fn, const uint32_t& Fd, const std::string& model) {
		uint32_t period = Fd / Fn;
		std::vector<double> signal_model (period);
		double t = 1. / Fd;
		if (model == "sin") {
				for (size_t i = 0; i < period; ++i) {
						signal_model[i] = A *sin(PI_2 * Fn * t * i);
				}
		} else {
				for (size_t i = 0; i < period; ++i) {
						signal_model[i] = A * sin(PI_2 * Fn * t * i + PI / 2);
				}
		}
		return signal_model;
}

std::complex<double> ComplexSignal(const std::vector<double>& signal, const std::vector<double>& model_is_sin, const std::vector<double>& model_is_cos) {
		double sum_is_sin = 0;
		double sum_is_cos = 0;
		for (size_t i = 0; i < signal.size(); ++i) {
				sum_is_sin += signal[i] * model_is_sin[i];
				sum_is_cos += signal[i] * model_is_cos[i];
		}
		std::complex<double> signal_complex (sum_is_cos, sum_is_sin);
		return signal_complex;
}

std::vector<uint32_t> Demodulation(const std::vector<double>& signal, const uint32_t& Fd, const double& Fn, const double& A,const uint32_t& block_size) {
		double t = 1. / Fd;
		uint32_t period = Fd / Fn;
		std::vector<uint32_t> bits (signal.size() / period - 1) ;
		std::vector<double> tmp_signal (period);
		std::vector<double> model_is_sin = SignalModel(A, Fn, Fd, "sin");
		std::vector<double> model_is_cos = SignalModel(A, Fn, Fd, "cos");
		uint32_t counter_1 = 0;
		uint32_t counter_2 = period;
		std::copy(signal.begin(), signal.begin() + period, tmp_signal.begin());
		std::complex<double> signal_complex_0_T = ComplexSignal(tmp_signal, model_is_sin, model_is_cos);
		for (size_t i = 0; i < bits.size(); ++i) {
					std::copy(signal.begin() + counter_1 + period, signal.begin() + counter_2 + period, tmp_signal.begin());
					std::complex<double> signal_complex_T_2T = ComplexSignal(tmp_signal, model_is_sin, model_is_cos);
					counter_1 += period;
					counter_2 += period;
					if (std::arg(signal_complex_0_T / signal_complex_T_2T) > -1.57079632679 && std::arg(signal_complex_0_T / signal_complex_T_2T) < 1.57079632679) {
							bits[i] = 0;
					} else {
							bits[i] = 1;
					}
					signal_complex_0_T = signal_complex_T_2T;
		}
		return bits;
}

std::vector <uint32_t> GenerateNumbers(const uint32_t& alphabet, const uint32_t& alphabet_length, std::vector<uint32_t> prefix, std::vector<uint32_t> arrays) { 
		if (alphabet_length == 0) { 
				for (const uint32_t& n : prefix) {
						arrays.push_back(n); 
				} 
        return arrays; 

    } 
    for (int digit = 0; digit < alphabet; digit++) { 
        prefix.push_back(digit); 
		arrays = GenerateNumbers(alphabet, alphabet_length - 1, prefix, arrays); 
	   	prefix.pop_back(); 
    } 
     return arrays; 
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

void AddNormalNoise(std::vector<double>& signal,const double& mean, const double& dispersion) {
	// АБГШ к сигналу
	std::mt19937 generator(rand());
	double standart_deviation = sqrt(dispersion);
	std::normal_distribution<double> distribution(mean,standart_deviation);
	for (double& sample : signal) {
			sample += distribution(generator);
	}
}

void HammingCode(const std::vector<uint32_t>& inf_bits, std::vector<uint32_t>& code_sequence) {
	for (size_t i = 0; i < inf_bits.size() / 4; ++i) {
			std::copy(inf_bits.begin() + i * 4, inf_bits.begin() + i * 4 + 4, code_sequence.begin() + i * 7);
			code_sequence[i * 7 + 4] = (inf_bits[i * 4] + inf_bits[i * 4 + 1] + inf_bits[i * 4 + 2]) % 2;
			code_sequence[i * 7 + 5] = (inf_bits[i * 4 + 1] + inf_bits[i * 4 + 2] + inf_bits[i * 4 + 3]) % 2;
			code_sequence[i * 7 + 6] = (inf_bits[i * 4] + inf_bits[i * 4 + 1] + inf_bits[i * 4 + 3]) % 2;
	}
}

void PossibleStates(std::vector<std::vector<uint32_t>>& all_states, const uint32_t& out_bits_size, const uint32_t& in_bits_size) {
		std::vector<uint32_t> prefix;
		std::vector<uint32_t> possible_states;
		possible_states = GenerateNumbers(2, in_bits_size, prefix, possible_states);
		std::vector<std::vector<uint32_t>> tmp (pow(2, in_bits_size), std::vector<uint32_t> (in_bits_size));
		VectorTo2dVector(tmp, possible_states);
		//HammingCode(tmp, all_states);
}

void HammingDecode(std::vector<uint32_t>& code_sequences, std::vector<uint32_t>& inf_bits) {
		std::map<std::vector<uint32_t>, uint32_t> syndroms = {{{1, 0, 1}, 0}, {{1, 1, 1}, 1}, {{1, 1, 0}, 2}, {{0, 1, 1}, 3}, {{1, 0, 0}, 4}, {{0, 1, 0}, 5}, {{0, 0, 1}, 6}};
		std::vector<uint32_t> syndrome (3);
		for (size_t i = 0; i < code_sequences.size() / 7; ++i) {
				syndrome[0] = (code_sequences[i * 7 + 4] + code_sequences[i * 7] + code_sequences[i * 7 + 1] + code_sequences[i * 7 + 2]) % 2;
				syndrome[1] = (code_sequences[i * 7 + 5] + code_sequences[i * 7 + 1] + code_sequences[i * 7 + 2] + code_sequences[i * 7 + 3]) % 2;
				syndrome[2] = (code_sequences[i * 7 + 6] + code_sequences[i * 7] + code_sequences[i * 7 + 1] + code_sequences[i * 7 + 3]) % 2;
				std::map<std::vector<uint32_t>, uint32_t>::iterator check = syndroms.find(syndrome);
				if (check != syndroms.end()) {
						code_sequences[i * 7 + check->second] = (code_sequences[i * 7 + check->second] + 1) % 2;
				}
				std::copy(code_sequences.begin() + i * 7, code_sequences.begin() + i * 7 + 4 , inf_bits.begin() + i * 4);
		}
}


double CheckError(const std::vector<uint32_t>& out_bits, const std::vector<uint32_t>& in_bits) {
	double  errs = 0;
	for (size_t i = 0; i < out_bits.size(); ++i) {
			errs += (out_bits[i] + in_bits[i]) % 2;
	}
	return errs / in_bits.size();
}

void Plotting(const uint32_t& count) {
	uint32_t block = 1000000;
	const double Fn = 1000;
	const double Fd = 10000;
	std::vector<double> A (count);
	std::vector<double> h (count);
	std::vector<double> err (count);
	const uint32_t period = Fd / Fn;
	const double mean = 0;
	const double dispersion = 1;
	double coef = 0.5;
	std::vector<uint32_t> out_bits (block * 4); // Информационные биты (кратны 4)
	for (uint32_t& bit : out_bits) {
		std::random_device rnd;
		std::mt19937 gen(rnd());
		std::uniform_int_distribution<uint32_t> number(0, 1);
		bit = number(gen);
	}
	std::vector<uint32_t> out_code_sequence (block * 7); // Код Хэмминга 7,4,3
	HammingCode(out_bits, out_code_sequence);
	for (size_t i = 0; i < count; ++i) {
		h[i] = i * coef;
		A[i] = sqrt(4 * h[i] * dispersion / period); 
		std::vector<double> signal = Modulation(out_bits, A[i], Fn, Fd);
		AddNormalNoise(signal, mean, dispersion);
		std::vector<uint32_t> in_code_sequence = Demodulation(signal, Fd, Fn, A[i], block); // Информационные биты (кратны 4)
		std::vector<uint32_t> in_bits (block * 4); // Информационные биты (кратны 4)
		HammingDecode(in_code_sequence, in_bits);
		err[i] = CheckError(out_bits, in_bits);

		std::cout << "Итерация: " << i << " Ошибка: " << err[i] << std::endl;
	}
//		WriteToTxt(h, "h_hamming.txt");
//		WriteToTxt(err, "err_hamming.txt");*/
}

int main () {
	Plotting(8);
	return 0;
}
