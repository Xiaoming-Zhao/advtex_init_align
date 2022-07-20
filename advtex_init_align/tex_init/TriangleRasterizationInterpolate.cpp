#include "TriangleRasterizationInterpolate.h"
#include <queue>

namespace NS_TriangleRI {

#define CoordCheck(u,v) {\
        const CoordFloat tmp_mtl_img_coord = current + (CoordFloat){u, v};\
        /* NOTE: we assume (x, y) are the center of each pixel, namely in the form of _.5 */ \
        const int ix = std::floor(tmp_mtl_img_coord.x) * mtlImg.width() + std::floor(tmp_mtl_img_coord.y);\
        float n1, n2;\
        /* NOTE: we allow each pixel to be discovered at most twice. */ \
        if(tmp_mtl_img_coord.isInImg(mtlImg.width(), mtlImg.height()) && localFlags[ix]==0 && tmp_mtl_img_coord.isInTriangle(mtlCoord1, mtl_dir1, mtl_dir2, invdet, n1, n2)) {\
            localFlags[ix] = 1; \
            q.push(tmp_mtl_img_coord);\
            /*if(debug) {*/ \
            /*    std::cout << "here:" << ix << " " << ((int)globalFlags[ix]) << std::endl;*/\
            /*}*/\
            /*if (globalFlags[ix] == -1) {*/\
            /*    globalFlags[ix] = textureIX;*/\
            const CoordFloat tmp_streamImgCoord = stream_img_dir1 * n1 + stream_img_dir2 * n2 + streamImgCoord1;\
            TriangleRasterizationInterpolate::CopyPixel(streamImg, mtlImg, tmp_mtl_img_coord, tmp_streamImgCoord, textureIX,\
                interpolator);\
            /*}*/\
        }\
}


void TriangleRasterizationInterpolate::CopyPixel(const boost::gil::rgb8c_view_t& streamImg, boost::gil::rgb8_view_t& mtlImg,
               const CoordFloat& mtl_img_coord, const CoordFloat& streamImgCoord, const int& textureIX, 
               const Interpolator interpolator) {
    
    // NOTE: we assume mtl coordinates (x, y) are the center of each pixel, namely in the form of _.5
    // So the pixel value will be in [0, max_val), pay attention to the interval ends
    int mtl_img_x = (int)std::floor(mtl_img_coord.x);
    int mtl_img_y = (int)std::floor(mtl_img_coord.y);


    // std::cout << "stream x: " << stream_img_x << "(" << streamImg.width() << "), y: " \
    //           << stream_img_y << "(" << streamImg.height() << ")" << std::endl;
    // std::cout << "mtl x: " << mtl_img_x << "(" << mtlImg.width() << "), y: " \
    //           << mtl_img_y << "(" << mtlImg.height() << ")" << std::endl << std::endl;
    
    if ((mtl_img_x >= 0) && (mtl_img_x < mtlImg.height()) && (mtl_img_y >= 0) && (mtl_img_y < mtlImg.width())) {

        // Use black and white to debug
        // mtlImg(mtl_img_x, mtl_img_y) = boost::gil::rgb8_pixel_t{255, 255, 255};

        // Use texture ID to debug
        // mtlImg(mtl_img_y, mtl_img_x) = boost::gil::rgb8_pixel_t{textureIX, textureIX, textureIX};

        mtlImg(mtl_img_y, mtl_img_x) = interpolator.interpolate(streamImgCoord);
    }
}

void TriangleRasterizationInterpolate::FillTriangle(const CoordFloat& mtlCoord1, const CoordFloat& mtlCoord2, const CoordFloat& mtlCoord3,
                  const CoordFloat& streamImgCoord1, const CoordFloat& streamImgCoord2, const CoordFloat& streamImgCoord3,
                  // std::vector<char>& globalFlags,
                  std::vector<char>& localFlags,
                  const int& textureIX,
                  const boost::gil::rgb8c_view_t& streamImg, boost::gil::rgb8_view_t& mtlImg) {
    
    /*
    Tricks:
    - we need float instead of int to obtain accurate coordinates wrt basis
    - we need to clearly specify our principle. Currently, I check the center of each pixel
    - local flags is a must-have
    - we need relaxation to mimic matlab's poly2mask approach
    - which pixel's RGB value to use? Currently using the RGB from first GLOBAL discovery.
      Matlab code uses the last GLOBAL discovery.
      I have tried both and do not find big difference.
    */
    //bool debug = false;

    const CoordFloat mtl_dir1 = mtlCoord2 - mtlCoord1;
    const CoordFloat mtl_dir2 = mtlCoord3 - mtlCoord1;
    const float invdet = 1.f / (float)(mtl_dir1.x * mtl_dir2.y - mtl_dir1.y * mtl_dir2.x);

    const CoordFloat stream_img_dir1 = streamImgCoord2 - streamImgCoord1;
    const CoordFloat stream_img_dir2 = streamImgCoord3 - streamImgCoord1;

    // std::vector<char> localFlags(mtlImg.height() * mtlImg.width(), 0);//this is expensive as a lot of memory needs to be allocated; move away?

    // initialize a clean map for localFlags
    // for (std::vector<char>::iterator it = localFlags.begin(); it != localFlags.end(); it++) *it = 0;
    std::fill(localFlags.begin(), localFlags.end(), 0);

    // auto start = std::chrono::high_resolution_clock::now();

    std::queue<CoordFloat> q;



    // NOTE: all three vertices need to be processed
    const CoordFloat mtlCoord1_pixel_center = {std::floor(mtlCoord1.x) + 0.5f, std::floor(mtlCoord1.y) + 0.5f};
    const CoordFloat mtlCoord2_pixel_center = {std::floor(mtlCoord2.x) + 0.5f, std::floor(mtlCoord2.y) + 0.5f};
    const CoordFloat mtlCoord3_pixel_center = {std::floor(mtlCoord3.x) + 0.5f, std::floor(mtlCoord3.y) + 0.5f};


    q.push(mtlCoord1_pixel_center);
    q.push(mtlCoord2_pixel_center);
    q.push(mtlCoord3_pixel_center);

    const Interpolator interpolator(streamImg, streamImgCoord1, streamImgCoord2, streamImgCoord3);

    while(!q.empty()) {

        CoordFloat current = q.front();

        // check the starting pixel itself
        // essentially, this check will be used only once
        CoordCheck(0, 0)

        CoordCheck(-1, -1)
        CoordCheck(-1, 0)
        CoordCheck(-1, 1)
        CoordCheck(0, -1)
        CoordCheck(0, 1)
        CoordCheck(1, -1)
        CoordCheck(1, 0)
        CoordCheck(1, 1)

        q.pop();
    }

    // auto finish = std::chrono::high_resolution_clock::now();
    // std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "ns" << std::endl << std::endl;
}

}