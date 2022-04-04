#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <map>


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

void HammingDecode(const std::vector<std::vector<uint32_t>>& code_sequence, std::vector<std::vector<uint32_t>>& inf_bits) {
	std::map<std::vector<uint32_t>, uint32_t> syndroms = {{{1, 0, 1}, 0}, {{1, 1, 1}, 1}, {{1, 1, 0}, 2}, {{0, 1, 1}, 3}, {{1, 0, 0}, 4}, {{0, 1, 0}, 5}, {{0, 0, 1}, 6}}; 
	std::vector<uint32_t> syndrome (3);
	for (size_t i = 0; i < code_sequence.size(); ++i) {
			syndrome[0] = (code_sequence[i][4] + code_sequence[i][0] + code_sequence[i][1] + code_sequence[i][2]) % 2;
			syndrome[1] = (code_sequence[i][5] + code_sequence[i][1] + code_sequence[i][2] + code_sequence[i][3]) % 2;
			syndrome[2] = (code_sequence[i][6] + code_sequence[i][0] + code_sequence[i][1] + code_sequence[i][3]) % 2;
			std::copy(code_sequence[i].begin(), code_sequence[i].begin() + inf_bits[0].size(), inf_bits[i].begin());
			std::map<std::vector<uint32_t>, uint32_t>::iterator check = syndroms.find(syndrome);
			if (check != syndroms.end()) {
					inf_bits[i][check->second] = (inf_bits[i][check->second] + 1) % 2;
			}

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

std::vector <uint32_t> generate_numbers(const uint32_t& alphabet, const uint32_t& alphabet_length, std::vector<uint32_t> prefix, std::vector<uint32_t> arrays) { 
		if (alphabet_length == 0) { 
				for (const uint32_t& n : prefix) {
						arrays.push_back(n); 
				} 
        return arrays; 

    } 
    for (int digit = 0; digit < alphabet; digit++) 
     { 
        prefix.push_back(digit); 
         arrays = generate_numbers(alphabet, alphabet_length - 1, prefix, arrays); 
         prefix.pop_back(); 
  
    } 
     return arrays; 
}


int main () {
	uint32_t block = 10;
	std::vector<std::vector<uint32_t>> inf_bits (block, std::vector<uint32_t> (4)); // Информационные биты (кратны 4)
	for (std::vector<uint32_t>& bits : inf_bits) {
		RandBits(bits);
	}
	Print2dVector(inf_bits);
	std::vector<std::vector<uint32_t>> code_sequence (block, std::vector<uint32_t> (7)); // Код Хэмминга 7,4,3
	std::cout <<'\n';
	HammingCode(inf_bits, code_sequence);
	Print2dVector(code_sequence);	
	std::vector<std::vector<uint32_t>> bits (block, std::vector<uint32_t> (4)); // Информационные биты (кратны 4)
	HammingDecode(code_sequence, bits);
	std::cout <<'\n';
	Print2dVector(bits);
	std::cout << CheckError(inf_bits, bits);
	return 0;
}
