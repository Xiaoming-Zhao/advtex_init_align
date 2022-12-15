#ifndef H_OURIO
#define H_OURIO

#include <vector>
#include <string>

namespace NS_OurIO {

class TextureIO {
public:
    template<typename T>
    static void readFileIntoVector(std::vector<T>& vec, const std::string& fn);

    template<typename T>
    static void writeVectorToFile(std::vector<T>& vec, const std::string& fn);
};

}//end namespace

#endif