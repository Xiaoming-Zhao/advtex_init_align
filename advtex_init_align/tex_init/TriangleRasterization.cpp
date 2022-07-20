#include <queue>
#include <iostream>

#include "TriangleRasterization.h"

namespace NS_TriangleR {

#define OldCoordCheck(u, v) {\
    const CoordFloat tmpMtlImgCoord = current + (CoordFloat){u, v};\
    /* NOTE: we assume (x, y) are the center of each pixel, namely in the form of _.5 */ \
    const int ix = std::floor(tmpMtlImgCoord.x) * mtlWidth + std::floor(tmpMtlImgCoord.y);\
    float n1, n2;\
    /* NOTE: we allow each pixel to be discovered at most twice. */ \
    if(tmpMtlImgCoord.isInImg(mtlWidth, mtlHeight) && localFlags[ix]==0 && tmpMtlImgCoord.isInTriangle(mtlCoord1, mtlDir1, mtlDir2, invdet, n1, n2)) {\
        localFlags[ix] = 1; \
        q.push(tmpMtlImgCoord);\
        if (flagTopOneAssignedMtl) { \
            const CoordFloat tmpStreamImgCoord = streamImgDir1 * n1 + streamImgDir2 * n2 + streamImgCoord1;\
            TriangleRasterization::CopyPixel(streamImg, mtlImg, tmpMtlImgCoord, tmpStreamImgCoord, textureIX);\
        } else { \
            TriangleRasterization::CopyTriLocalCoords(mtlTriLocalCoords1, mtlTriLocalCoords2, tmpMtlImgCoord, n1, n2);\
        } \
        /*}*/\
    }\
}


#define ExtrudeCoordCheck(curMtlImgCoord) { \
    float extrudeN1, extrudeN2; \
    /* int nExtrudePixels = 5; */ \
    for (int extrudeU = -1 * nExtrudePixels; extrudeU < nExtrudePixels + 1; extrudeU++) { \
        for (int extrudeV = -1 * nExtrudePixels; extrudeV < nExtrudePixels + 1; extrudeV++) { \
            if ((extrudeU != 0) && (extrudeV != 0)) { \
                const CoordFloat tmpExtrudeMtlImgCoord = curMtlImgCoord + (CoordFloat){(float)extrudeU, (float)extrudeV}; \
                const int extrudeIx = std::floor(tmpExtrudeMtlImgCoord.x) * mtlWidth + std::floor(tmpExtrudeMtlImgCoord.y);\
                if (globalFlags[extrudeIx] == 0 && tmpExtrudeMtlImgCoord.isInImg(mtlWidth, mtlHeight) && !tmpExtrudeMtlImgCoord.isInTriangle(mtlCoord1, mtlDir1, mtlDir2, invdet, extrudeN1, extrudeN2)) { \
                    if (flagTopOneAssignedMtl) { \
                        const CoordFloat tmpExtrudeStreamImgCoord = streamImgDir1 * extrudeN1 + streamImgDir2 * extrudeN2 + streamImgCoord1; \
                        TriangleRasterization::CopyPixel(streamImg, mtlImg, tmpExtrudeMtlImgCoord, tmpExtrudeStreamImgCoord, textureIX); \
                    } else { \
                        TriangleRasterization::CopyTriLocalCoords(mtlTriLocalCoords1, mtlTriLocalCoords2, tmpExtrudeMtlImgCoord, extrudeN1, extrudeN2);\
                    } \
                } \
            } \
        } \
    } \
}


#define CoordCheck(u, v) {\
    const CoordFloat tmpMtlImgCoord = current + (CoordFloat){u, v};\
    /* NOTE: we assume (x, y) are the center of each pixel, namely in the form of _.5 */ \
    const int ix = std::floor(tmpMtlImgCoord.x) * mtlWidth + std::floor(tmpMtlImgCoord.y);\
    float n1, n2;\
    /* NOTE: we allow each pixel to be discovered at most twice. */ \
    if(tmpMtlImgCoord.isInImg(mtlWidth, mtlHeight) && localFlags[ix] == 0 && tmpMtlImgCoord.isInTriangle(mtlCoord1, mtlDir1, mtlDir2, invdet, n1, n2)) {\
        localFlags[ix] = 1; \
        globalFlags[ix] = 1; \
        q.push(tmpMtlImgCoord);\
        if (flagTopOneAssignedMtl) { \
            const CoordFloat tmpStreamImgCoord = streamImgDir1 * n1 + streamImgDir2 * n2 + streamImgCoord1;\
            TriangleRasterization::CopyPixel(streamImg, mtlImg, tmpMtlImgCoord, tmpStreamImgCoord, textureIX);\
        } else { \
            TriangleRasterization::CopyTriLocalCoords(mtlTriLocalCoords1, mtlTriLocalCoords2, tmpMtlImgCoord, n1, n2);\
        } \
        /* extrude by several pixel */ \
        ExtrudeCoordCheck(tmpMtlImgCoord); \
    }\
}


void TriangleRasterization::CopyPixel(const boost::gil::rgb8c_view_t& streamImg, boost::gil::rgb8_view_t& mtlImg,
                                      const CoordFloat& mtlImgCoord, const CoordFloat& streamImgCoord, const int& textureIX)
{
    
    // NOTE: we assume mtl coordinates (x, y) are the center of each pixel, namely in the form of _.5
    // So the pixel value will be in [0, max_val), pay attention to the interval ends
    int mtlImgX = (int)std::floor(mtlImgCoord.x);
    int mtlImgY = (int)std::floor(mtlImgCoord.y);

    int streamImgX = (int)std::floor(streamImgCoord.x);
    int streamImgY = (int)std::floor(streamImgCoord.y);
    
    if ((mtlImgX >= 0) && (mtlImgX < mtlImg.height()) && (mtlImgY >= 0) && (mtlImgY < mtlImg.width())) {
        // Use black and white to debug
        // mtlImg(mtlImgX, mtlImgY) = boost::gil::rgb8_pixel_t{255, 255, 255};

        // Use texture ID to debug
        // mtlImg(mtlImgY, mtlImgX) = boost::gil::rgb8_pixel_t{textureIX, textureIX, textureIX};

        if ((streamImgX >= 0) && (streamImgX < streamImg.height()) && (streamImgY >= 0) && (streamImgY < streamImg.width())) {
            // NOTE: boost::gil's 1st coordiante is for horizontal
            mtlImg(mtlImgY, mtlImgX) = streamImg(streamImgY, streamImgX); //This is where copying happens
        }
    }
}


void TriangleRasterization::CopyTriLocalCoords(Eigen::MatrixXf& mtlTriLocalCoords1, Eigen::MatrixXf& mtlTriLocalCoords2,
                                               const CoordFloat& mtlImgCoord, const float& n1, const float& n2)
{
    // NOTE: we assume mtl coordinates (x, y) are the center of each pixel, namely in the form of _.5
    // So the pixel value will be in [0, max_val), pay attention to the interval ends
    int mtlImgX = (int)std::floor(mtlImgCoord.x);
    int mtlImgY = (int)std::floor(mtlImgCoord.y);

    if ((mtlImgX >= 0) && (mtlImgX < mtlTriLocalCoords1.rows()) && (mtlImgY >= 0) && (mtlImgY < mtlTriLocalCoords1.cols())) {
        mtlTriLocalCoords1(mtlImgX, mtlImgY) = n1;
        mtlTriLocalCoords2(mtlImgX, mtlImgY) = n2;
    }
}


void TriangleRasterization::FillTriangle(const CoordFloat& mtlCoord1, const CoordFloat& mtlCoord2, const CoordFloat& mtlCoord3,
                                         const CoordFloat& streamImgCoord1, const CoordFloat& streamImgCoord2, const CoordFloat& streamImgCoord3,
                                         // std::vector<char>& globalFlags,
                                         std::vector<char>& localFlags, std::vector<char>& globalFlags,
                                         const int& textureIX,
                                         const boost::gil::rgb8c_view_t& streamImg, boost::gil::rgb8_view_t& mtlImg,
                                         Eigen::MatrixXf& mtlTriLocalCoords1, Eigen::MatrixXf& mtlTriLocalCoords2,
                                         const bool& flagTopOneAssignedMtl, const int& nExtrudePixels)
{
    
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

    int mtlWidth = mtlImg.width();
    int mtlHeight = mtlImg.height();
    if (!flagTopOneAssignedMtl) {
        mtlWidth = mtlTriLocalCoords1.cols();
        mtlHeight = mtlTriLocalCoords1.rows();
    }

    const CoordFloat mtlDir1 = mtlCoord2 - mtlCoord1;
    const CoordFloat mtlDir2 = mtlCoord3 - mtlCoord1;
    const float invdet = 1.f / (float)(mtlDir1.x * mtlDir2.y - mtlDir1.y * mtlDir2.x);

    const CoordFloat streamImgDir1 = streamImgCoord2 - streamImgCoord1;
    const CoordFloat streamImgDir2 = streamImgCoord3 - streamImgCoord1;

    std::fill(localFlags.begin(), localFlags.end(), 0);

    // auto start = std::chrono::high_resolution_clock::now();

    std::queue<CoordFloat> q;

    // NOTE: all three vertices need to be processed
    const CoordFloat mtlPixelCenterCoord1 = {std::floor(mtlCoord1.x) + 0.5f, std::floor(mtlCoord1.y) + 0.5f};
    const CoordFloat mtlPixelCenterCoord2 = {std::floor(mtlCoord2.x) + 0.5f, std::floor(mtlCoord2.y) + 0.5f};
    const CoordFloat mtlPixelCenterCoord3 = {std::floor(mtlCoord3.x) + 0.5f, std::floor(mtlCoord3.y) + 0.5f};

    q.push(mtlPixelCenterCoord1);
    q.push(mtlPixelCenterCoord2);
    q.push(mtlPixelCenterCoord3);

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
}

}