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

double LikelihoodRatio(const double& phi, const uint32_t& bit, const double& d, const double& A) {
    double bes = std::cyl_bessel_i(0, d);
	double phi_0 = 1 / (PI_2 * bes) * exp(d * cos(phi));
	double phi_1 = 1 / (PI_2 * bes) * exp(d * cos(PI - std::abs(phi)));
    if (bit == 0) {
        return  (phi_0 / phi_1) / (1 + (phi_0 / phi_1));
    }
    return 1 / (1 + phi_0 / phi_1);
}


std::vector<double> Modulation(std::vector<uint32_t> bits,const double& A, const double& Fn, const double& Fd) {
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

std::vector<uint32_t> Demodulation(const std::vector<double>& signal, const uint32_t& Fd, const double& Fn, const double& A, const uint32_t& block_size, const double& d, std::vector<double>& probability_of_received_bits) {
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
					probability_of_received_bits[i] = LikelihoodRatio(std::arg(signal_complex_0_T / signal_complex_T_2T), bits[i], d, A);
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

std::vector<std::vector<uint32_t>> PossibleStates() {
		std::vector<std::vector<uint32_t>> all_states (16, std::vector<uint32_t> (7));
		std::vector<uint32_t> prefix;
		std::vector<uint32_t> possible_states;
		possible_states = GenerateNumbers(2, 4, prefix, possible_states);
		std::vector<uint32_t> tmp (16 * 7);
		HammingCode(possible_states, tmp);
		VectorTo2dVector(all_states, tmp);
		return all_states;
}

double SequenceProbability(const std::vector<uint32_t>& code_sequence, const std::vector<double>& probability_of_received_bits, const std::vector<uint32_t> state) {
    double probability = 1;
    for (size_t i = 0; i < code_sequence.size(); ++i) {
            if (code_sequence[i] == state[i]) {
                    probability *= probability_of_received_bits[i];
            } else {
                    probability *= 1 - probability_of_received_bits[i];
            }
    }
    return probability;
}

void HammingDecode(std::vector<uint32_t>& code_sequences, std::vector<uint32_t>& inf_bits, const std::vector<double>& probability_of_received_bits) {
		std::vector<uint32_t> tmp_code_sequences (7);
		std::vector<double> tmp_probability_of_received_bits (7);
		std::vector<std::vector<uint32_t>> all_states = PossibleStates();
		std::vector<double> sequences_probability (all_states.size());
		for (size_t i = 0; i < code_sequences.size() / 7; ++i) {
				std::copy(code_sequences.begin() + i * 7, code_sequences.begin() + i * 7 + 7, tmp_code_sequences.begin());
				std::copy(probability_of_received_bits.begin() + i * 7, probability_of_received_bits.begin() + i * 7 + 7, tmp_probability_of_received_bits.begin());
				for (size_t j = 0; j < all_states.size(); ++j) {
						sequences_probability[j] = SequenceProbability(tmp_code_sequences, tmp_probability_of_received_bits, all_states[j]);
				}
		        uint32_t element_with_maximum_probability = std::distance(sequences_probability.begin(), (std::max_element(sequences_probability.begin(), sequences_probability.end())));
        		std::copy(all_states[element_with_maximum_probability].begin(), all_states[element_with_maximum_probability].begin() + 4, inf_bits.begin() + i * 4);
		}
}


double CheckError(const std::vector<uint32_t>& out_bits, const std::vector<uint32_t>& in_bits) {
	double  errs = 0;
	for (size_t i = 0; i < out_bits.size(); ++i) {
			errs += (out_bits[i] + in_bits[i]) % 2;
	}
	return errs / in_bits.size();
}

void Plotting() {
	uint32_t block = 1000000;
	const double Fn = 1000;
	const double Fd = 10000;
	std::vector<double> d = {0, 1.5, 4.35, 9.25};
	std::vector<double> A = {0, 1, 2, 3};
	std::vector<double> h (A.size());
	std::vector<double> err (A.size());
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
	for (size_t i = 0; i < A.size(); ++i) {
		h[i] = pow(A[i], 2) * period / 4 / dispersion;
		std::vector<double> signal = Modulation(out_code_sequence, A[i], Fn, Fd);
		AddNormalNoise(signal, mean, dispersion);
		std::vector<double> probability_of_received_bits (block * 7);
		std::vector<uint32_t> in_code_sequence = Demodulation(signal, Fd, Fn, A[i], block, d[i], probability_of_received_bits); // Информационные биты (кратны 4)
		std::vector<uint32_t> in_bits (block * 4); // Информационные биты (кратны 4)
		HammingDecode(in_code_sequence, in_bits, probability_of_received_bits);
		err[i] = CheckError(out_bits, in_bits);
		std::cout << "Итерация: " << i << " Ошибка: " << err[i] << std::endl;
	}
		WriteToTxt(h, "my_h_hamming_ofm.txt");
		WriteToTxt(err, "my_err_hamming_ofm.txt");
}

int main () {
	Plotting();
	return 0;
}
