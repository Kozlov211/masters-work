#include <cmath>
#include <iostream>
#include <ostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <complex>


using namespace std;

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

double LikelihoodRatio(const double& phi, const uint32_t& bit, const double& d, const double& A) {
    double bes = std::cyl_bessel_i(0, d);
	double phi_0 = 1 / (PI_2 * bes) * exp(d * cos(phi));
	double phi_1 = 1 / (PI_2 * bes) * exp(d * cos(PI - std::abs(phi)));
    if (bit == 0) {
        return  (phi_0 / phi_1) / (1 + (phi_0 / phi_1));
    }
    return 1 / (1 + phi_0 / phi_1);
}

double SequenceProbability(const std::vector<uint32_t>& code_sequence, const std::vector<double>& bit_reliability, const std::vector<uint32_t> state) {
    double probability = 1;
    for (size_t i = 0; i < code_sequence.size(); ++i) {
    	if (code_sequence[i] == state[i]) {
        	probability *= bit_reliability[i];
        } else {
            probability *= 1 - bit_reliability[i];
        }
    }
    return probability;
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

std::vector<std::vector<uint32_t>> PossibleStates(const uint32_t& out_bits_size, const uint32_t& in_bits_size) {
	std::vector<uint32_t> prefix;
	std::vector<uint32_t> possible_states;
	possible_states = GenerateNumbers(2, in_bits_size, prefix, possible_states);
	std::vector<std::vector<uint32_t>> tmp (pow(2, in_bits_size), std::vector<uint32_t> (in_bits_size));
	VectorTo2dVector(tmp, possible_states);
	std::vector<std::vector<uint32_t>> all_states = Coding(tmp);
	for (std::vector<uint32_t>& states : all_states) {
		states.erase(states.begin());
	}
	return all_states;
}

std::vector<std::vector<uint32_t>> Demodulation (const std::vector<std::vector<std::complex<double>>>& signal_with_noise, std::vector<std::vector<double>>& bit_reliability, const double& d, const double& A) {
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
			bit_reliability[i][j] = LikelihoodRatio(phi, in_bits[i][j], d, A);
			signal_complex_0_T = signal_complex_T_2T;

		}		
	}
	return in_bits;
}


std::vector<std::vector<uint32_t>> Decoding(const std::vector<std::vector<uint32_t>>& in_code_sequences , const std::vector<std::vector<double>>& probability_of_received_bits) {
	std::vector<std::vector<uint32_t>> in_bits (in_code_sequences.size(), std::vector<uint32_t> (4));
	std::vector<std::vector<uint32_t>> all_states = PossibleStates(in_code_sequences[0].size(), in_bits[0].size());
	std::vector<double> sequences_probability (all_states.size());
	for (size_t i = 0; i < in_code_sequences.size(); ++i) {
		for (size_t j = 0; j < all_states.size(); ++j) {
			sequences_probability[j] = SequenceProbability(in_code_sequences[i], probability_of_received_bits[i], all_states[j]);
		}
		cout << "Вероятность кодовой комбинации" << endl;
		PrintVector(sequences_probability);
		uint32_t element_with_maximum_probability = std::distance(sequences_probability.begin(), (std::max_element(sequences_probability.begin(), sequences_probability.end())));
		std::copy(all_states[element_with_maximum_probability].begin(), all_states[element_with_maximum_probability].begin() + in_bits[0].size(), in_bits[i].begin());
	}
	return in_bits;
}

bool ModTwoAddVectors(const std::vector<uint32_t>& vec1, const std::vector<uint32_t> vec2) {
	for (size_t i = 0; i < vec1.size(); ++i) {
			if (vec1[i] != vec2[i]) {
				return true;
			}
	}
	return false;
}

double CheckError(const std::vector<std::vector<uint32_t>>& out_bits, const std::vector<std::vector<uint32_t>>& in_bits) {
	double errs = 0;
	for (size_t i = 0; i < out_bits.size(); ++i) {
			errs += ModTwoAddVectors(out_bits[i], in_bits[i]);
	}
	return errs / static_cast<double>(in_bits.size());
}

int main () {
	uint32_t block = 1;
	std::vector<std::vector<uint32_t>> out_bits (block, std::vector<uint32_t> (4));
	for (std::vector<uint32_t>& bits : out_bits) {
		RandBits(bits);
	}
	std::vector<std::vector<uint32_t>> out_code_sequences = Coding(out_bits);
	std::vector<double> A = {0, sqrt(2), 2, sqrt(6)};
	std::vector<double> d = {0, 1.5, 4.35, 9.25};
	std::vector<double> errs (4);
	cout << "Кодовая комбинация" << endl;
	Print2dVector(out_code_sequences);
	for (size_t i = 0; i < A.size(); ++i) {
		cout << "Отношение сигнал/шум: " << i << endl;
		std::vector<std::vector<std::complex<double>>> signal =  Modulation(out_code_sequences, A[i]);
		std::vector<std::vector<std::complex<double>>> signal_with_noise = AddNormalNoise(signal);
		std::vector<std::vector<double>> bit_reliability (block, std::vector<double> (7));
		std::vector<std::vector<uint32_t>> in_code_sequences = Demodulation(signal_with_noise, bit_reliability, d[i], A[i]);
		cout << "Надежность битов" << endl;
		Print2dVector(bit_reliability);
		std::vector<std::vector<uint32_t>> in_bits = Decoding(in_code_sequences, bit_reliability);
		errs[i] = CheckError(out_bits, in_bits);
		cout << endl;
//		std::cout << CheckError(out_bits, in_bits) << std::endl;
	}
//	WriteToTxt(errs, "errs_vector_my.txt");
//	std::vector<std::vector<std::complex<double>>> signal =  Modulation(out_code_sequences, A);
//	std::vector<std::vector<std::complex<double>>> signal_with_noise = AddNormalNoise(signal);
//	std::vector<std::vector<double>> bit_reliability (block, std::vector<double> (7));
//	std::vector<std::vector<uint32_t>> in_code_sequences = Demodulation(signal_with_noise, bit_reliability, d, A);
//	std::vector<std::vector<uint32_t>> in_bits = Decoding(in_code_sequences, bit_reliability);
//	std::cout << CheckError(out_bits, in_bits) << std::endl;
//	Print2dVector(out_code_sequences);
//	Print2dVector(in_code_sequences);
//	Print2dVector(out_bits);
//	Print2dVector(in_bits);
	return 0;
}
