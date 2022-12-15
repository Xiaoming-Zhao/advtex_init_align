#ifndef H_STREAM_FILE
#define H_STREAM_FILE

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <boost/gil/image.hpp>
#include <boost/gil/typedefs.hpp>

#include "ImgWriter.h"


namespace NS_StreamFile {

#define APPLE_MAX_DEPTH 5.0f
#define APPLE_DEPTH_WIDTH 192
#define APPLE_DEPTH_HEIGHT 256

typedef Eigen::Map<const Eigen::Matrix4f> MapMatrix4f;   // column major map
typedef Eigen::Map<const Eigen::MatrixXf> MapMatrixXf;
enum StreamTypeEnum {STREAM_UNKNOWN, STREAM_APPLE, STREAM_COLMAP};

class StreamFile {
private:
    struct ImInfo {
        unsigned int height;
        unsigned int width;
        unsigned int realheight;
    };
    struct RGBInfo {
        unsigned int height;
        unsigned int width;
        unsigned int channels;
    };
    struct DInfo {
        unsigned int height;
        unsigned int width;
    };

    StreamTypeEnum streamType;

    size_t nEntries;

    std::string fn;

    std::vector<std::streampos> streamYOffset, streamCBCROffset, streamRGBOffset;
    std::vector<ImInfo> YImgSizes;
    std::vector<ImInfo> CBCRImgSizes;
    std::vector<RGBInfo> RGBImgSizes;
    std::vector<boost::gil::rgb8_image_t*> RGBImages;

    std::ifstream depthfid;
    std::vector<std::streampos> streamDepthOffset;
    std::vector<DInfo> DepthSizes;
    std::vector<std::vector<float>> depthStorage;
    std::vector<MapMatrixXf> depthMaps;

    std::vector<float> matrixStorage;
    std::vector<MapMatrix4f> viewMatrixMap, projMatrixMap;
    Eigen::Matrix<float, Eigen::Dynamic, 4> viewMatrix, projMatrix, transformMatrix;

    boost::gil::rgb8_image_t* LoadStreamImage(int IX) const;
    boost::gil::rgb8_image_t* LoadStreamImageApple(int IX) const;
    boost::gil::rgb8_image_t* LoadStreamImageCOLMAP(int IX) const;
    void IndexStreamApple(std::ifstream& ifs);
    void IndexStreamCOLMAP(std::ifstream& ifs);
public:
    StreamFile() : streamType(STREAM_UNKNOWN) {}
    ~StreamFile();
    void ClearData();

    enum StreamTypeEnum GetStreamType() const;

    bool IndexFile(const std::string& fn, const int& globalDebugFlag, StreamTypeEnum streamtype);

    size_t NumberOfEntries() const;

    const Eigen::Matrix<float, Eigen::Dynamic, 4>& GetViewMatrix() const;
    const Eigen::Matrix<float, Eigen::Dynamic, 4>& GetProjectionMatrix() const;
    const Eigen::Matrix<float, Eigen::Dynamic, 4>& GetTransformMatrix() const;
    void WriteCameraMatrixBinary(const std::string& fileCameraMatrix);

    bool StartDepthQuerySequence();
    bool DepthQuery(const int IX, const std::streamoff& row, const std::streamoff& col, float& depth);
    void EndDepthQuerySequence();
    void DepthSizeQuery(int& height, int& width) const;
    void ReadAllDepthMaps();
    bool DepthQueryFromMem(const int IX, const std::streamoff& row, const std::streamoff& col, float& depth) const;
    void SaveDepthMapToDisk(const std::string& binFile) const;

    const boost::gil::rgb8_image_t* const GetRGBImage(const int IX);
    void ReadAllRGBImages();
    void RGBSizeQuery(const int& viewID, float& height, float& width) const;
    void SaveRGBToDisk(NS_ImgWriter::ImgWriter& imgWriter, const std::string& rawDir) const;
};

} //end namespace

#endif