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

double LikelihoodRatio(const double& phi, const uint32_t& bit, const double& dispersion) {
	double standart_deviation = sqrt(dispersion);
	double phi_0 = 1 / (standart_deviation * sqrt(PI_2)) * exp(-1 / (2 * standart_deviation) * pow((phi - 0), 2));
	double phi_1 = 1 / (standart_deviation * sqrt(PI_2)) * exp(-1 / (2 * standart_deviation) * pow((std::abs(phi) - PI), 2));
	if (bit == 0) {
	 	return  (phi_0 / phi_1) / (1 + (phi_0 / phi_1));
	}
	return 1 / (1 + phi_0 / phi_1);
}

std::vector<double> Modulation(const std::vector<std::vector<uint32_t>>& bits,const double& A, const double& Fn, const double& Fd) {
		uint32_t period = Fd / Fn;
		double t = 1. / Fd;
		std::vector<double> signal (period * bits.size() * bits[0].size());
		for (size_t i = 0; i < bits.size(); ++i) {
				for (size_t j = 0; j < bits[0].size(); ++j)
						if (bits[i][j] == 0) {
							for (size_t k = 0 ; k < period; ++k) {
									signal[(i * bits[0].size() + j) * period + k] = A * sin(PI_2 * k * t * Fn);
							}
						} else {
							for (size_t k = 0 ; k < period; ++k) {
									signal[(i * bits[0].size() + j) * period + k] = A * sin(PI_2 * k * t * Fn + PI);
							}
						}
		}
		return signal;
}

std::vector<double> SignalModel(const double& Fn, const uint32_t& Fd, const std::string& model) {
		uint32_t period = Fd / Fn;
		std::vector<double> signal_model (period);
		double t = 1. / Fd;
		if (model == "sin") {
				for (size_t i = 0; i < period; ++i) {
						signal_model[i] = sin(PI_2 * Fn * t * i);
				}
		} else {
				for (size_t i = 0; i < period; ++i) {
						signal_model[i] = sin(PI_2 * Fn * t * i + PI / 2);
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
		std::complex<double> signal_complex (sum_is_sin, sum_is_cos);
		return signal_complex;
}

std::vector<std::vector<uint32_t>> Demodulation(const std::vector<double>& signal, const uint32_t& Fd, const double& Fn, const uint32_t& block_size, const double& dispersion, std::vector<std::vector<double>>& probability_of_received_bits) {
		double t = 1. / Fd;
		uint32_t period = Fd / Fn;
		std::vector<std::vector<uint32_t>> bits (block_size, std::vector<uint32_t> (signal.size() / block_size / period));
		std::vector<uint32_t> tmp_bits (signal.size() / period);
		std::vector<double> tmp_probability (tmp_bits.size());
		std::vector<double> tmp_signal (period);
		std::vector<double> model_is_sin = SignalModel(Fn, Fd, "sin");
		std::vector<double> model_is_cos = SignalModel(Fn, Fd, "cos");
		uint32_t counter_1 = 0;
		uint32_t counter_2 = period;
		for (size_t i = 0; i < tmp_bits.size(); ++i) {
				std::copy(signal.begin() + counter_1, signal.begin() + counter_2, tmp_signal.begin());
				std::complex<double> signal_complex = ComplexSignal(tmp_signal, model_is_sin, model_is_cos);
				counter_1 += period;
				counter_2 += period;
				if (std::arg(signal_complex) > -1.57079632679 && std::arg(signal_complex) < 1.57079632679) {
						tmp_bits[i] = 0;
				} else {
						tmp_bits[i] = 1;
				}
				tmp_probability[i] = LikelihoodRatio(std::arg(signal_complex), tmp_bits[i], dispersion);
		}
		VectorTo2dVector(bits, tmp_bits);
		VectorTo2dVector(probability_of_received_bits, tmp_probability);
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

void HammingCode(const std::vector<std::vector<uint32_t>>& inf_bits, std::vector<std::vector<uint32_t>>& code_sequence) {
	for (size_t i = 0; i < inf_bits.size(); ++i) {
			std::copy(inf_bits[i].begin(), inf_bits[i].end(), code_sequence[i].begin());
			code_sequence[i][4] = (inf_bits[i][0] + inf_bits[i][1] + inf_bits[i][2]) % 2;
			code_sequence[i][5] = (inf_bits[i][1] + inf_bits[i][2] + inf_bits[i][3]) % 2;
			code_sequence[i][6] = (inf_bits[i][0] + inf_bits[i][1] + inf_bits[i][3]) % 2;
	}
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

void PossibleStates(std::vector<std::vector<uint32_t>>& all_states, const uint32_t& out_bits_size, const uint32_t& in_bits_size) {
		std::vector<uint32_t> prefix;
		std::vector<uint32_t> possible_states;
		possible_states = GenerateNumbers(2, in_bits_size, prefix, possible_states);
		std::vector<std::vector<uint32_t>> tmp (pow(2, in_bits_size), std::vector<uint32_t> (in_bits_size));
		VectorTo2dVector(tmp, possible_states);
		HammingCode(tmp, all_states);
}

void HammingDecode(const std::vector<std::vector<uint32_t>>& code_sequences, std::vector<std::vector<uint32_t>>& inf_bits, const std::vector<std::vector<double>>& probability_of_received_bits) {
	std::vector<std::vector<uint32_t>> all_states (pow(2, inf_bits[0].size()), std::vector<uint32_t> (code_sequences[0].size()));
	PossibleStates(all_states, code_sequences[0].size(), inf_bits[0].size());
	std::vector<double> sequences_probability (all_states.size());
	for (size_t i = 0; i < code_sequences.size(); ++i) {
		for (size_t j = 0; j < all_states.size(); ++j) {
			sequences_probability[j] = SequenceProbability(code_sequences[i], probability_of_received_bits[i], all_states[j]);
		}
		uint32_t element_with_maximum_probability = std::distance(sequences_probability.begin(), (std::max_element(sequences_probability.begin(), sequences_probability.end())));
		std::copy(all_states[element_with_maximum_probability].begin(), all_states[element_with_maximum_probability].begin() + inf_bits[0].size(), inf_bits[i].begin());
	}
}

uint32_t ModTwoAddVectors(const std::vector<uint32_t>& vec1, const std::vector<uint32_t> vec2) {
	uint32_t result = 0;
	for (size_t i = 0; i < vec1.size(); ++i) {
			result += (vec1[i] + vec2[i]) % 2;
	}
	return result;
}

double CheckError(const std::vector<std::vector<uint32_t>>& out_bits, const std::vector<std::vector<uint32_t>>& in_bits) {
	double  errs = 0;
	for (size_t i = 0; i < out_bits.size(); ++i) {
			errs += ModTwoAddVectors(out_bits[i], in_bits[i]);
	}
	return errs / in_bits.size() / in_bits[0].size();
}

void Plotting(const uint32_t& count) {
	uint32_t block = 2500000;
	const double Fn = 1000;
	const double Fd = 10000;
	std::vector<double> A (count);
	std::vector<double> h (count);
	std::vector<double> err (count);
	const uint32_t period = Fd / Fn;
	const double mean = 0;
	const double dispersion = 1;
	double coef = 0.5;
	for (double i = 0; i < count; ++i) {
			A[i] = sqrt(4 * i * coef * dispersion / period); 
			h[i] = i * coef;
	}
	std::vector<std::vector<uint32_t>> out_bits (block, std::vector<uint32_t> (4)); // Информационные биты (кратны 4)
	for (std::vector<uint32_t>& bits : out_bits) {
		RandBits(bits);
	}
	std::vector<std::vector<uint32_t>> out_code_sequence (block, std::vector<uint32_t> (7)); // Код Хэмминга 7,4,3
	HammingCode(out_bits, out_code_sequence);
	for (size_t i = 0; i < count; ++i) {
		std::vector<double> signal = Modulation(out_code_sequence, A[i], Fn, Fd);
		AddNormalNoise(signal, mean, dispersion);
		std::vector<std::vector<double>> probability_of_received_bits (block, std::vector<double> (out_code_sequence[0].size()));
		std::vector<std::vector<uint32_t>> in_code_sequence = Demodulation(signal, Fd, Fn, block, dispersion, probability_of_received_bits); // Информационные биты (кратны 4)
		std::vector<std::vector<uint32_t>> in_bits (block, std::vector<uint32_t> (4)); // Информационные биты (кратны 4)
		HammingDecode(in_code_sequence, in_bits, probability_of_received_bits);
		err[i] = CheckError(out_bits, in_bits);
		std::cout << err[i] << std::endl;
	}
	WriteToTxt(h, "h.txt");
	WriteToTxt(err, "err.txt");
}

int main () {
	Plotting(20);
	return 0;
}
