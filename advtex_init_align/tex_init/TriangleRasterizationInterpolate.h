#ifndef H_TRIANGLE_RASTERIZATION_INTERPOLATE
#define H_TRIANGLE_RASTERIZATION_INTERPOLATE

#include <vector>
#include <iostream>

#include <boost/gil/image.hpp>
#include <boost/gil/typedefs.hpp>
#include "TriangleRasterization.h"

namespace NS_TriangleRI {

typedef NS_TriangleR::CoordFloat CoordFloat;

// This could also be a class, but I wasn't sure if there 
// were any overhead differences
struct Interpolator {
    float denom;
    const CoordFloat& c1, c2, c3;
    boost::uint8_t r1, r2, r3;
    boost::uint8_t g1, g2, g3;
    boost::uint8_t b1, b2, b3;

    //Interpolator(const CoordFloat& c1, const CoordFloat& c2, const CoordFloat& c3,
    //            const boost::gil::rgb8_pixel_t& p1, const boost::gil::rgb8_pixel_t& p2, const boost::gil::rgb8_pixel_t& p3): c1(c1), c2(c2), c3(c3) {
    Interpolator(const boost::gil::rgb8c_view_t& streamImg, const CoordFloat& c1,
            const CoordFloat& c2, const CoordFloat& c3): c1(c1), c2(c2), c3(c3) {
        int img_x1 = (int)std::floor(c1.x);
        int img_y1 = (int)std::floor(c1.y);
        int img_x2 = (int)std::floor(c2.x);
        int img_y2 = (int)std::floor(c2.y);
        int img_x3 = (int)std::floor(c3.x);
        int img_y3 = (int)std::floor(c3.y);

        const boost::gil::rgb8_pixel_t p1 = streamImg(img_y1, img_x1);
        const boost::gil::rgb8_pixel_t p2 = streamImg(img_y2, img_x2);
        const boost::gil::rgb8_pixel_t p3 = streamImg(img_y3, img_x3);
        denom = (c2.y - c3.y)*(c1.x - c3.x) + (c3.x - c2.x)*(c1.y - c3.y);
        boost::gil::red_t R;
        boost::gil::green_t G;
        boost::gil::blue_t B;
        r1 = boost::gil::get_color(p1, R);
        r2 = boost::gil::get_color(p2, R);
        r3 = boost::gil::get_color(p3, R);
        g1 = boost::gil::get_color(p1, G);
        g2 = boost::gil::get_color(p2, G);
        g3 = boost::gil::get_color(p3, G);
        b1 = boost::gil::get_color(p1, B);
        b2 = boost::gil::get_color(p2, B);
        b3 = boost::gil::get_color(p3, B);
    } 

    inline boost::gil::rgb8_pixel_t interpolate(const CoordFloat& p) const {
        // There are terms for w1/w2 that could be precomputed and saved,
        // but I don't know enough to know if this actually saves time
        float w1 = ((c2.y - c3.y)*(p.x - c3.x) + (c3.x - c2.x)*(p.y - c3.y)) / denom;
        float w2 = ((c3.y - c1.y)*(p.x - c3.x) + (c1.x - c3.x)*(p.y - c3.y)) / denom;
        float w3 = 1 - w1 - w2;
        boost::uint8_t r = r1*w1 + r2*w2 + r3*w3;
        boost::uint8_t g = g1*w1 + g2*w2 + g3*w3;
        boost::uint8_t b = b1*w1 + b2*w2 + b3*w3;
        return boost::gil::rgb8_pixel_t(r, g, b);
    }

};

class TriangleRasterizationInterpolate {
public:
    static void FillTriangle(const CoordFloat& mtlCoord1, const CoordFloat& mtlCoord2, const CoordFloat& mtlCoord3,
                  const CoordFloat& streamImgCoord1, const CoordFloat& streamImgCoord2, const CoordFloat& streamImgCoord3,
                  // std::vector<char>& globalFlags,
                  std::vector<char>& localFlags, const int& textureIX,
                  const boost::gil::rgb8c_view_t& streamImg, boost::gil::rgb8_view_t& mtlImg);
    
private:
    static void CopyPixel(const boost::gil::rgb8c_view_t& streamImg, boost::gil::rgb8_view_t& mtlImg,
               const CoordFloat& mtl_img_coord, const CoordFloat& streamImgCoord, const int& textureIX, 
               const Interpolator interpolator);

    static boost::gil::rgb8_pixel_t Interpolate(const boost::gil::rgb8_pixel_t& p1,
               const boost::gil::rgb8_pixel_t& p2, const boost::gil::rgb8_pixel_t& p3,
               const float& n1, const float& n2);

};

}//end namespace
#endif