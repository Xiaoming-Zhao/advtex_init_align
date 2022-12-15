#ifndef H_TRIANGLE_RASTERIZATION_AVERAGE
#define H_TRIANGLE_RASTERIZATION_AVERAGE

#include <vector>
#include <iostream>

#include <boost/gil/image.hpp>
#include <boost/gil/typedefs.hpp>
#include "TriangleRasterization.h"

namespace NS_TriangleRA {

typedef NS_TriangleR::CoordFloat CoordFloat;

// This could also be a class, but I wasn't sure if there 
// were any overhead differences
struct Accumulator {
    int count;
    float r, g, b;

    //Interpolator(const CoordFloat& c1, const CoordFloat& c2, const CoordFloat& c3,
    //            const boost::gil::rgb8_pixel_t& p1, const boost::gil::rgb8_pixel_t& p2, const boost::gil::rgb8_pixel_t& p3): c1(c1), c2(c2), c3(c3) {
    Accumulator(): count(0), r(0), g(0), b(0) {}

    inline void accumulate(const boost::gil::rgb8_pixel_t& p) {
        count++;
        float rnew = (float)boost::gil::get_color(p, boost::gil::red_t());
        float gnew = (float)boost::gil::get_color(p, boost::gil::green_t());
        float bnew = (float)boost::gil::get_color(p, boost::gil::blue_t());
        if (count == 1) {
            r = rnew;
            g = gnew;
            b = bnew;
        } else {
            r = r + (rnew - r) / count;
            g = g + (gnew - g) / count;
            b = b + (bnew - b) / count;
        }
    }

    inline boost::gil::rgb8_pixel_t get_mean() {
        return boost::gil::rgb8_pixel_t(
            (boost::uint8_t)r,
            (boost::uint8_t)g,
            (boost::uint8_t)b
        );
    }


};

class TriangleRasterizationAverage {
public:
    static void FillTriangle(const CoordFloat& mtlCoord1, const CoordFloat& mtlCoord2, const CoordFloat& mtlCoord3,
                  const CoordFloat& streamImgCoord1, const CoordFloat& streamImgCoord2, const CoordFloat& streamImgCoord3,
                  // std::vector<char>& globalFlags,
                  std::vector<char>& localFlags, const int& textureIX,
                  const boost::gil::rgb8c_view_t& streamImg, boost::gil::rgb8_view_t& mtlImg);
    
private:
    static void AccumulatePixel(const boost::gil::rgb8c_view_t& streamImg,
               const CoordFloat& streamImgCoord, Accumulator& accumulator);
    static void FillPixel(boost::gil::rgb8_view_t& mtlImg,
               const CoordFloat& mtlImgCoord, boost::gil::rgb8_pixel_t& mean_p);


};

}//end namespace
#endif