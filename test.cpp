#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
using namespace std;


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

int main()
{
    // spot check for ν == 0i
	vector<double> bes_x (10);
	for (size_t i = 0; i < bes_x.size(); ++i) {
		bes_x[i] = std::cyl_bessel_i(0, i * 0.5);
	} 
 	WriteToTxt(bes_x, "bes.txt");

}
