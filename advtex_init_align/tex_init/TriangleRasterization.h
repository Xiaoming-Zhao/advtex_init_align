#ifndef H_TRIANGLE_RASTERIZATION
#define H_TRIANGLE_RASTERIZATION

#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <boost/gil/image.hpp>
#include <boost/gil/typedefs.hpp>

namespace NS_TriangleR {

struct CoordFloat {
    // NOTE: x is for vertical (height) while y is for horizontal (width)
    float x, y;

    inline bool isInImg(const int Width, const int Height) const {
        return x >= 0 && x < Height && y >= 0 && y < Width;
    }
    inline bool isInTriangle(const CoordFloat& base, const CoordFloat& dir1, const CoordFloat& dir2,
                             const float invdet, float& n1, float& n2) const {
        const CoordFloat pt_dir = *this - base;
        n1 = (dir2.y * pt_dir.x - dir2.x * pt_dir.y) * invdet;
        n2 = (-dir1.y * pt_dir.x + dir1.x * pt_dir.y) * invdet;
        return (n1 >= 0) && (n2 >= 0) && (n1 + n2 <= 1);

    }

    inline CoordFloat operator+(const CoordFloat& o) const {
        return {this->x + o.x, this->y + o.y};
    }
    inline CoordFloat operator-(const CoordFloat& o) const {
        return {this->x - o.x, this->y - o.y};
    }
    inline CoordFloat operator*(const float& o) const {
        return {this->x * o, this->y * o};
    }
};

class TriangleRasterization {
public:
    static void FillTriangle(const CoordFloat& mtlCoord1, const CoordFloat& mtlCoord2, const CoordFloat& mtlCoord3,
                             const CoordFloat& streamImgCoord1, const CoordFloat& streamImgCoord2, const CoordFloat& streamImgCoord3,
                             // std::vector<char>& globalFlags,
                             std::vector<char>& localFlags, std::vector<char>& globalFlags, const int& textureIX,
                             const boost::gil::rgb8c_view_t& streamImg, boost::gil::rgb8_view_t& mtlImg,
                             Eigen::MatrixXf& mtlTriLocalCoords1, Eigen::MatrixXf& mtlTriLocalCoords2,
                             const bool& flagTopOneAssignedMtl = true, const int& nExtrudePixels = 3);
    
private:
    static void CopyPixel(const boost::gil::rgb8c_view_t& streamImg, boost::gil::rgb8_view_t& mtlImg,
                          const CoordFloat& mtl_img_coord, const CoordFloat& streamImgCoord, const int& textureIX);
    static void CopyTriLocalCoords(Eigen::MatrixXf& mtlTriLocalCoords1, Eigen::MatrixXf& mtlTriLocalCoords2,
                                   const CoordFloat& mtlImgCoord, const float& n1, const float& n2);

};

}//end namespace
#endif