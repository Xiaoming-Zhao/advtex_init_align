#include "OurIO.h"

#include <fstream>
#include <iostream>

namespace NS_OurIO {

template<typename T>
void TextureIO::readFileIntoVector(std::vector<T>& vec, const std::string& fn) {
    std::ifstream fh(fn, std::ios_base::binary | std::ios_base::in | std::ios_base::ate);
    if(!fh.is_open()) {
        std::cout << "Couldn't find file " << fn << std::endl;
        return;
    }
    vec.resize(fh.tellg()/sizeof(T));
    fh.seekg(fh.beg);
    fh.read((char*)&vec[0], vec.size()*sizeof(T));
    fh.close();
}

template<typename T>
void TextureIO::writeVectorToFile(std::vector<T>& vec, const std::string& fn) {
    std::ofstream ofs(fn, std::ios_base::binary | std::ios_base::out);
    if(!ofs.is_open()) {
        std::cout << "Error opening file " << fn << std::endl;
        return;
    }
    ofs.write((char*)&vec[0], vec.size()*sizeof(T));
    ofs.close();
}

template void TextureIO::readFileIntoVector<float>(std::vector<float>& vec, const std::string& fn);
template void TextureIO::readFileIntoVector<double>(std::vector<double>& vec, const std::string& fn);
template void TextureIO::readFileIntoVector<unsigned int>(std::vector<unsigned int>& vec, const std::string& fn);

}//end namespace