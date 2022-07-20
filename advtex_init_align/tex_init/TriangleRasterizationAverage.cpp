#include "TriangleRasterizationAverage.h"
#include <queue>

namespace NS_TriangleRA {

#define CoordCheckAccumulate(u,v) {\
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
            TriangleRasterizationAverage::AccumulatePixel(streamImg, tmp_streamImgCoord, accumulator);\
            /*}*/\
        }\
}


#define CoordCheckFill(u,v) {\
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
            TriangleRasterizationAverage::FillPixel(mtlImg, tmp_mtl_img_coord, mean_p);\
            /*}*/\
        }\
}


void TriangleRasterizationAverage::AccumulatePixel(const boost::gil::rgb8c_view_t& streamImg,
               const CoordFloat& streamImgCoord, Accumulator& accumulator) {
    
    // NOTE: we assume mtl coordinates (x, y) are the center of each pixel, namely in the form of _.5
    // So the pixel value will be in [0, max_val), pay attention to the interval ends
    int stream_img_x = (int)std::floor(streamImgCoord.x);
    int stream_img_y = (int)std::floor(streamImgCoord.y);

    if ((stream_img_x >= 0) && (stream_img_x < streamImg.height()) && (stream_img_y >= 0) && (stream_img_y < streamImg.width())) {
        accumulator.accumulate(streamImg(stream_img_y, stream_img_x));
    }
}

void TriangleRasterizationAverage::FillPixel(boost::gil::rgb8_view_t& mtlImg,
               const CoordFloat& mtlImgCoord, boost::gil::rgb8_pixel_t& mean_p) {
    
    // NOTE: we assume mtl coordinates (x, y) are the center of each pixel, namely in the form of _.5
    // So the pixel value will be in [0, max_val), pay attention to the interval ends
    int mtl_img_x = (int)std::floor(mtlImgCoord.x);
    int mtl_img_y = (int)std::floor(mtlImgCoord.y);

    if ((mtl_img_x >= 0) && (mtl_img_x < mtlImg.height()) && (mtl_img_y >= 0) && (mtl_img_y < mtlImg.width())) {
        mtlImg(mtl_img_y, mtl_img_x) = mean_p;
    }
}


void TriangleRasterizationAverage::FillTriangle(const CoordFloat& mtlCoord1, const CoordFloat& mtlCoord2, const CoordFloat& mtlCoord3,
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

    Accumulator accumulator;
    while(!q.empty()) {

        CoordFloat current = q.front();

        // check the starting pixel itself
        // essentially, this check will be used only once
        CoordCheckAccumulate(0, 0)

        CoordCheckAccumulate(-1, -1)
        CoordCheckAccumulate(-1, 0)
        CoordCheckAccumulate(-1, 1)
        CoordCheckAccumulate(0, -1)
        CoordCheckAccumulate(0, 1)
        CoordCheckAccumulate(1, -1)
        CoordCheckAccumulate(1, 0)
        CoordCheckAccumulate(1, 1)

        q.pop();
    }
    boost::gil::rgb8_pixel_t mean_p = accumulator.get_mean();
    std::fill(localFlags.begin(), localFlags.end(), 0);

    q.push(mtlCoord1_pixel_center);
    q.push(mtlCoord2_pixel_center);
    q.push(mtlCoord3_pixel_center);
    while(!q.empty()) {

        CoordFloat current = q.front();

        // check the starting pixel itself
        // essentially, this check will be used only once
        CoordCheckFill(0, 0)

        CoordCheckFill(-1, -1)
        CoordCheckFill(-1, 0)
        CoordCheckFill(-1, 1)
        CoordCheckFill(0, -1)
        CoordCheckFill(0, 1)
        CoordCheckFill(1, -1)
        CoordCheckFill(1, 0)
        CoordCheckFill(1, 1)

        q.pop();
    }
    // auto finish = std::chrono::high_resolution_clock::now();
    // std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "ns" << std::endl << std::endl;
}

}