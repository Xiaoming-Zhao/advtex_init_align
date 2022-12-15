#ifndef H_UTILS
#define H_UTILS

#include <iostream>
#include <fstream>
#include <random>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <boost/gil/image.hpp>
#include <boost/gil/typedefs.hpp>
#include <boost/gil/extension/io/png.hpp>

#include "MPSolver.h"

namespace NS_Utils {

#define FLOAT_PRECISION 1e-5f

#define ASSERT(condition) { if(!(condition)){ std::cerr << "ASSERT FAILED: " << #condition << " @ " << __FILE__ << " (" << __LINE__ << ")" << std::endl; } }

int ArgPos(char* str, int argc, char **argv);

bool CheckLineSegIntersect2D(const float& x1, const float& y1, const float& x2, const float& y2,
                             const float& x3, const float& y3, const float& x4, const float& y4);

struct Tuple {
        unsigned int k;  // model index
        unsigned int f;  // traingle/face index

        Tuple(unsigned int k, unsigned int f) : k(k), f(f) {}
        // OurTuple(const OurTuple& t) {k = t.k; f = t.f;}
    };

struct Triple {
    int c[3];
    const int length = 3;
    friend std::ostream& operator<<(std::ostream& os, const Triple& o) {
        os << o.c[0] << " " << o.c[1] << " " << o.c[2];
        return os;
    }
};

// this is used for checking common two elements in a pair of Triple
const int nCombine = 18;
const int combination[18][4] = {
    {0, 1, 0, 1}, {0, 1, 1, 0},
    {0, 1, 0, 2}, {0, 1, 2, 0},
    {0, 1, 1, 2}, {0, 1, 2, 1},
    {0, 2, 0, 1}, {0, 2, 1, 0},
    {0, 2, 0, 2}, {0, 2, 2, 0},
    {0, 2, 1, 2}, {0, 2, 2, 1},
    {1, 2, 0, 1}, {1, 2, 1, 0},
    {1, 2, 0, 2}, {1, 2, 2, 0},
    {1, 2, 1, 2}, {1, 2, 2, 1}};

bool GetTwoCommonElementsInTriple(const Triple& triple1, const Triple& triple2, int& elem1, int& elem2);


void UVCoordsFromTriangle3D(const Eigen::Vector3d& pt1, const Eigen::Vector3d pt2, const Eigen::Vector3d pt3,
                            Eigen::Vector2d& uv1, Eigen::Vector2d& uv2, Eigen::Vector2d& uv3,
                            Eigen::Vector2d& minUV, Eigen::Vector2d& maxUV);

void MakeDir(const std::string dir);

template <typename T>
std::vector<size_t> getSortedIdxes(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  std::stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}


void WriteMatrixBin(const Eigen::MatrixXf& Mat, std::ofstream& outFileStream, const bool& orderC);


void PlotNDC(boost::gil::rgb8_image_t* img, Eigen::MatrixXf& NDC, const int& ID);

void TestMPSolver(NS_MPSolver::MPSolver<float, int>& MP, std::default_random_engine& generator, std::uniform_real_distribution<float>& uniformDist);

} //end namespace NS_Utils

#endif /* ifndef H_UTILS */