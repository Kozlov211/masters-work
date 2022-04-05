#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <map>
#include <cmath>
#include <algorithm>
template <typename Type>
void PrintVector(const std::vector<Type>& S) {
	// Печать вектора в консоль
	for (const Type& num : S) {
		std::cout << num << std::endl;
	}
}

template <typename Type> 
void Print2dVector(const std::vector<std::vector<Type>>& vec) {
	for (const std::vector<Type>& nums : vec) {
		for (const Type& num : nums) {
			std::cout << num << "\t";
		}
		std::cout << std::endl;
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

void ProbabilityOfReceivedBits(std::vector<double>& probability_of_received_bits) {
	std::random_device rnd;
	std::mt19937 gen(rnd());
	std::uniform_real_distribution<double> result(0, 1);
	for (double& element : probability_of_received_bits) {
			element = result(gen);
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

void HammingDecode(const std::vector<std::vector<uint32_t>>& code_sequences, std::vector<std::vector<uint32_t>>& inf_bits) {
	std::vector<uint32_t> prefix;
	std::vector<uint32_t> tmp;
	tmp = GenerateNumbers(2, code_sequences[0].size(), prefix, tmp);
	std::vector<std::vector<uint32_t>> all_states (pow(2, code_sequences[0].size()), std::vector<uint32_t> (code_sequences[0].size()));
	VectorTo2dVector(all_states, tmp);
	std::vector<double> probability_of_received_bits (code_sequences[0].size());
	std::vector<double> sequences_probability (pow(2, code_sequences.size()));
	ProbabilityOfReceivedBits(probability_of_received_bits);
	for (size_t i = 0; i < code_sequences.size(); ++i) {
		for (size_t j = 0; j < code_sequences.size(); ++j) {
			sequences_probability[j] = SequenceProbability(code_sequences[i], probability_of_received_bits, all_states[j]);
		}
		uint32_t element_with_maximum_probability = std::distance(sequences_probability.begin(), (std::max_element(sequences_probability.begin(), sequences_probability.end())));
		inf_bits[i] = all_states[element_with_maximum_probability];
	}
}

uint32_t ModTwoAddVectors(const std::vector<uint32_t>& vec1, const std::vector<uint32_t> vec2) {
	uint32_t result = 0;
	for (size_t i = 0; i < vec1.size(); ++i) {
			result += (vec1[i] + vec2[i]) % 2;
	}
	return result;
}

uint32_t CheckError(const std::vector<std::vector<uint32_t>>& out_bits, const std::vector<std::vector<uint32_t>>& in_bits) {
	uint32_t errs = 0;
	for (size_t i = 0; i < out_bits.size(); ++i) {
			errs += ModTwoAddVectors(out_bits[i], in_bits[i]);
	}
	return errs;
}

int main () {
	uint32_t block = 10;
	std::vector<std::vector<uint32_t>> inf_bits (block, std::vector<uint32_t> (4)); // Информационные биты (кратны 4)
	for (std::vector<uint32_t>& bits : inf_bits) {
		RandBits(bits);
	}
	std::vector<std::vector<uint32_t>> code_sequence (block, std::vector<uint32_t> (7)); // Код Хэмминга 7,4,3
	HammingCode(inf_bits, code_sequence);
	std::vector<std::vector<uint32_t>> bits (block, std::vector<uint32_t> (4)); // Информационные биты (кратны 4)
	HammingDecode(code_sequence, bits);
	return 0;
}
