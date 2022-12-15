#include "StreamFile.h"

#include <iostream>
#include <chrono>

#include <boost/gil/image.hpp>
#include <boost/gil/typedefs.hpp>
#include <boost/gil/extension/io/png.hpp>
#include <boost/gil/extension/io/png.hpp>
#include <boost/gil/extension/numeric/sampler.hpp>
#include <boost/gil/extension/numeric/resample.hpp>
#include <boost/gil/extension/toolbox/color_spaces/ycbcr.hpp>

#include "Utils.h"

namespace NS_StreamFile {

StreamFile::~StreamFile() {
    ClearData();
}

void StreamFile::ClearData() {
    EndDepthQuerySequence();

    std::vector<std::streampos>().swap(streamYOffset);
    std::vector<std::streampos>().swap(streamCBCROffset);
    std::vector<std::streampos>().swap(streamDepthOffset);
    std::vector<std::streampos>().swap(streamRGBOffset);

    std::vector<StreamFile::ImInfo>().swap(YImgSizes);
    std::vector<StreamFile::ImInfo>().swap(CBCRImgSizes);
    std::vector<StreamFile::RGBInfo>().swap(RGBImgSizes);

    std::vector<StreamFile::DInfo>().swap(DepthSizes);
    std::vector<std::vector<float>>().swap(depthStorage);
    std::vector<MapMatrixXf>().swap(depthMaps);

    std::vector<float>().swap(matrixStorage);
    std::vector<MapMatrix4f>().swap(viewMatrixMap);
    std::vector<MapMatrix4f>().swap(projMatrixMap);

    for(int k = 0; k < RGBImages.size(); ++k) {
        delete RGBImages[k];
        RGBImages[k] = NULL;
    }
    std::vector<boost::gil::rgb8_image_t*>().swap(RGBImages);
    streamType = STREAM_UNKNOWN;
}

size_t StreamFile::NumberOfEntries() const {
    return nEntries;
}


void StreamFile::IndexStreamApple(std::ifstream& ifs) {
    std::vector<float> matrixData(4 * 4 * 2, 0.0f);
    StreamFile::ImInfo imInfo;
    while(true) {
        ifs.read((char*)&imInfo, sizeof(imInfo));
        if(ifs.eof()) break;
        YImgSizes.push_back(imInfo);
        RGBImgSizes.push_back(RGBInfo{imInfo.height, imInfo.width, 3});

        std::streampos buf = ifs.tellg();
        ifs.seekg(imInfo.width*imInfo.height, std::ios_base::cur);
        if(!ifs.eof()) streamYOffset.emplace_back(buf); else break;

        ifs.read((char*)&imInfo, sizeof(imInfo));
        if(ifs.eof()) break;
        CBCRImgSizes.push_back(imInfo);

        buf = ifs.tellg();
        ifs.seekg(imInfo.width*imInfo.height, std::ios_base::cur);
        if(!ifs.eof()) streamCBCROffset.emplace_back(buf); else break;

        buf = ifs.tellg();
        ifs.seekg(APPLE_DEPTH_WIDTH * APPLE_DEPTH_HEIGHT * 4, std::ios_base::cur);
        if(!ifs.eof()) streamDepthOffset.emplace_back(buf); else break;
        DepthSizes.push_back({APPLE_DEPTH_HEIGHT, APPLE_DEPTH_WIDTH});

        ifs.read((char*)&matrixData[0], matrixData.size()*sizeof(float));
        //std::copy_n(std::istream_iterator<float>(ifs), 4*4*2, std::back_inserter(matrixData));
        if(ifs.eof()) break;
        matrixStorage.insert(matrixStorage.end(), matrixData.begin(), matrixData.end());
    }
}


void StreamFile::IndexStreamCOLMAP(std::ifstream& ifs) {
    std::vector<float> matrixData(4 * 4 * 2, 0.0f);
    StreamFile::RGBInfo rgbInfo;
    StreamFile::DInfo dInfo;
    while(true) {
        ifs.read((char*)&rgbInfo, sizeof(rgbInfo));
        if(ifs.eof()) break;
        RGBImgSizes.push_back(rgbInfo);

        std::streampos buf = ifs.tellg();
        ifs.seekg(rgbInfo.width * rgbInfo.height * rgbInfo.channels, std::ios_base::cur);
        if(!ifs.eof()) streamRGBOffset.emplace_back(buf); else break;

        ifs.read((char*)&dInfo, sizeof(dInfo));
        if(ifs.eof()) break;
        DepthSizes.push_back(dInfo);

        buf = ifs.tellg();
        ifs.seekg(dInfo.width * dInfo.height * 4, std::ios_base::cur);
        if(!ifs.eof()) streamDepthOffset.emplace_back(buf); else break;

        ifs.read((char*)&matrixData[0], matrixData.size()*sizeof(float));
        //std::copy_n(std::istream_iterator<float>(ifs), 4*4*2, std::back_inserter(matrixData));
        if(ifs.eof()) break;
        matrixStorage.insert(matrixStorage.end(), matrixData.begin(), matrixData.end());
    }
}


bool StreamFile::IndexFile(const std::string& fn, const int& globalDebugFlag, StreamTypeEnum streamtype) {
    auto start = std::chrono::high_resolution_clock::now();

    std::ifstream ifs(fn, std::ios_base::binary | std::ios_base::in);
    if(!ifs.is_open()) {
        std::cout << "Could not open " << fn << std::endl;
        return false;
    }

    ClearData();
    this->fn = fn;
    streamType = streamtype;

    switch(streamtype) {
        case STREAM_APPLE:
            IndexStreamApple(ifs);
            break;
        case STREAM_COLMAP:
            IndexStreamCOLMAP(ifs);
            break;
        case STREAM_UNKNOWN:
        default:
            ifs.close();
            std::cout << "Unknown stream type." << std::endl;
            return false;
    }

    ifs.close();

    for(int k = 0; k < matrixStorage.size(); k += 4 * 4 * 2) {
        // column major, fill 1st column, then 2nd column, ...
        viewMatrixMap.push_back(MapMatrix4f(&matrixStorage[k]));
        projMatrixMap.push_back(MapMatrix4f(&matrixStorage[k + 4 * 4]));
    }

    nEntries = viewMatrixMap.size();

    viewMatrix.conservativeResize(nEntries * 4, Eigen::NoChange);
    projMatrix.conservativeResize(nEntries * 4, Eigen::NoChange);
    transformMatrix.conservativeResize(nEntries * 4, Eigen::NoChange);
    for(int k = 0; k < nEntries; ++k) {
        viewMatrix.block<4, 4>(4 * k, 0) = viewMatrixMap[k];
        projMatrix.block<4, 4>(4 * k, 0) = projMatrixMap[k];
        transformMatrix.block<4, 4>(4 * k, 0) = projMatrixMap[k] * viewMatrixMap[k];
    }

    RGBImages.resize(nEntries, NULL);

    std::cout << "Complete reading stream file." << std::endl;
    if (globalDebugFlag == 1) {
        std::cout << "[Debug] #cameras: " << NumberOfEntries() << std::endl;
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "Complete indexing stream file in " \
              << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() << " us" << std::endl;

    return true;
}


boost::gil::rgb8_image_t* StreamFile::LoadStreamImage(int IX) const {
    switch(streamType) {
        case STREAM_APPLE:
            return LoadStreamImageApple(IX);
            break;
        case STREAM_COLMAP:
            return LoadStreamImageCOLMAP(IX);
            break;
        case STREAM_UNKNOWN:
        default:
            std::cout << "Unknown stream type." << std::endl;
            return NULL;
    }
}


boost::gil::rgb8_image_t* StreamFile::LoadStreamImageCOLMAP(int IX) const {
    const StreamFile::RGBInfo& rgbSizes = RGBImgSizes[IX];
    std::vector<unsigned char> RGBImg(rgbSizes.height*rgbSizes.width*rgbSizes.channels, 0);
    std::ifstream fid(fn, std::ios_base::binary | std::ios_base::in);
    if(!fid.is_open()) {
        std::cout << "Could not open " << fn << std::endl;
        return NULL;
    }
    fid.seekg(streamRGBOffset[IX], std::ios_base::beg);
    fid.read((char*)&RGBImg[0], RGBImg.size()*sizeof(char));
    fid.close();

    boost::gil::rgb8_image_t* rgb = new boost::gil::rgb8_image_t(rgbSizes.width, rgbSizes.height);
    auto vrgb = boost::gil::view(*rgb);
    unsigned int offset = rgbSizes.width*rgbSizes.height;
    for(int x = 0; x < rgbSizes.width; ++x) {
        for(int y = 0; y < rgbSizes.height; ++y) {
            vrgb(x, y) = boost::gil::rgb8_pixel_t{RGBImg[x*rgbSizes.height+y], RGBImg[offset+x*rgbSizes.height+y], RGBImg[2*offset+x*rgbSizes.height+y]};
        }
    }
    return rgb;
}


boost::gil::rgb8_image_t* StreamFile::LoadStreamImageApple(int IX) const {
    const StreamFile::ImInfo& ySizes = YImgSizes[IX];
    const StreamFile::ImInfo& cbcrSizes = CBCRImgSizes[IX];

    std::vector<unsigned char> YImg(ySizes.height*ySizes.width, 0);
    std::vector<unsigned char> CBCRImg(cbcrSizes.height*cbcrSizes.width, 0);

    std::ifstream fid(fn, std::ios_base::binary | std::ios_base::in);
    if(!fid.is_open()) {
        std::cout << "Could not open " << fn << std::endl;
        return NULL;
    }
    fid.seekg(streamYOffset[IX], std::ios_base::beg);
    fid.read((char*)&YImg[0], YImg.size()*sizeof(char));
    fid.seekg(streamCBCROffset[IX], std::ios_base::beg);
    fid.read((char*)&CBCRImg[0], CBCRImg.size()*sizeof(char));
    fid.close();

    boost::gil::gray8_image_t cbImg(cbcrSizes.width, cbcrSizes.realheight);
    boost::gil::gray8_image_t crImg(cbcrSizes.width, cbcrSizes.realheight);
    auto vcb = boost::gil::view(cbImg);
    auto vcr = boost::gil::view(crImg);
    unsigned char* ptr = &CBCRImg[0];
    for(int x = 0; x < cbcrSizes.width; ++x) {
        for(int y = 0; y < cbcrSizes.realheight; ++y) {
            vcb(x,y) = boost::gil::gray8_pixel_t{CBCRImg[x*cbcrSizes.height + 2*y]};
            vcr(x,y) = boost::gil::gray8_pixel_t{CBCRImg[x*cbcrSizes.height + 2*y + 1]};
        }
    }
    //boost::gil::write_view("test1_0.png", boost::gil::const_view(cbImg), boost::gil::png_tag());
    //boost::gil::write_view("test2_0.png", boost::gil::const_view(crImg), boost::gil::png_tag());

    boost::gil::gray8_image_t cbImgReal(ySizes.width, ySizes.realheight);
    boost::gil::gray8_image_t crImgReal(ySizes.width, ySizes.realheight);

    boost::gil::resize_view(boost::gil::const_view(cbImg), boost::gil::view(cbImgReal), boost::gil::bilinear_sampler());
    boost::gil::resize_view(boost::gil::const_view(crImg), boost::gil::view(crImgReal), boost::gil::bilinear_sampler());
    //boost::gil::write_view("test1_1.png", boost::gil::const_view(cbImgReal), boost::gil::png_tag());
    //boost::gil::write_view("test2_1.png", boost::gil::const_view(crImgReal), boost::gil::png_tag());

    boost::gil::ycbcr_601_8_image_t ycbcrImg(ySizes.width, ySizes.realheight);
    auto v_ycbcr = boost::gil::view(ycbcrImg);
    auto v_cb = boost::gil::view(cbImgReal);
    auto v_cr = boost::gil::view(crImgReal);
    for(int x = 0; x < ySizes.width; ++x) {
        for(int y = 0; y < ySizes.realheight; ++y) {
            v_ycbcr(x,y) = boost::gil::ycbcr_601_8_pixel_t{YImg[y+x*ySizes.height],v_cb(x,y),v_cr(x,y)};
        }
    }

    boost::gil::rgb8_image_t* rgb = new boost::gil::rgb8_image_t(ySizes.width, ySizes.realheight);
    boost::gil::copy_and_convert_pixels(boost::gil::const_view(ycbcrImg), boost::gil::view(*rgb));
    //boost::gil::write_view("test3_1.png", boost::gil::const_view(*rgb), boost::gil::png_tag());
    return rgb;
}


void StreamFile::ReadAllRGBImages()
{
    std::cout << "start reading all RGB images ..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NumberOfEntries(); i++) {
        RGBImages[i] = LoadStreamImage(i);

        std::cout << i << std::endl;

    }


    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "... complete reading " << NumberOfEntries() << " RGB images in " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms" << std::endl;
}


const boost::gil::rgb8_image_t* const StreamFile::GetRGBImage(const int IX)
{
    if(IX >= NumberOfEntries()) {
        std::cout << "Invalid index" << std::endl;
        return NULL;
    }
    if  (RGBImages[IX] == NULL) {
        //load YCBCR and convert to RGB
        RGBImages[IX] = LoadStreamImage(IX);
    }
    return RGBImages[IX];
}


void StreamFile::RGBSizeQuery(const int& viewID, float& height, float& width) const
{
    height = float(RGBImgSizes[viewID].height);
    width = float(RGBImgSizes[viewID].width);
}



void StreamFile::SaveRGBToDisk(NS_ImgWriter::ImgWriter& imgWriter, const std::string& rawDir) const
{
    for (int i = 0; i < NumberOfEntries(); i++) {
        assert (RGBImages[i] != NULL);
        const boost::gil::rgb8_image_t* const rgbImg = RGBImages[i];
        std::string rawImgFile = rawDir + "/raw_" + std::to_string(i) + ".png";
        imgWriter.AddImageToQueue(*rgbImg, rawImgFile, NS_ImgWriter::ImgProcessTypeEnum::NONE);
    }
}


void StreamFile::ReadAllDepthMaps()
{
    std::cout << "start reading all depth maps ..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    depthfid.open(fn, std::ios_base::binary | std::ios_base::in);
    if(!depthfid.is_open()) {
        std::cout << "Error opening " << fn << std::endl;
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < NumberOfEntries(); i++) {
        std::vector<float> tmpStorage(DepthSizes[i].height * DepthSizes[i].width, 0.0f);

        std::streamoff offset = streamDepthOffset[i];
        depthfid.seekg(offset, std::ios_base::beg);
        depthfid.read((char*)&tmpStorage[0], DepthSizes[i].height * DepthSizes[i].width * sizeof(float));

        depthStorage.push_back(tmpStorage);
        // column major: filling 1st column, 2nd column, ...
        depthMaps.push_back(MapMatrixXf(&depthStorage[i][0], DepthSizes[i].height, DepthSizes[i].width));

        assert (depthStorage.size() == i + 1);
        assert (depthMaps.size() == i + 1);
    }

    depthfid.close();

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "... complete reading " << NumberOfEntries() << " depth maps in " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms" << std::endl;
}


void StreamFile::SaveDepthMapToDisk(const std::string& binFile) const
{
    std::cout << "start saving all depth maps ..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    std::ofstream outFileStream(binFile, std::ios_base::binary | std::ios_base::out);
    if (!outFileStream.is_open()) {
        std::cout << "Unable to open binary file for storing depth maps." << std::endl;
        exit(EXIT_FAILURE);
    }

    float rows, cols;
    for (int k = 0; k < depthMaps.size(); k++) {
        if (k == 0) {
            rows = (float)depthMaps[0].rows();
            cols = (float)depthMaps[0].cols();
            outFileStream.write((char*)&rows, sizeof(float));
            outFileStream.write((char*)&cols, sizeof(float));
        } else {
            // TODO: remove assumption that all depth maps have same resolution.
            assert (depthMaps[k].rows() == rows);
            assert (depthMaps[k].cols() == cols);
        }
        NS_Utils::WriteMatrixBin(depthMaps[k], outFileStream, true);
    }

    outFileStream.close();

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "... complete saving " << NumberOfEntries() << " depth maps to disk in " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms" << std::endl;
}


bool StreamFile::DepthQueryFromMem(const int IX, const std::streamoff& row, const std::streamoff& col, float& depth) const
{
    if (IX >= NumberOfEntries()) {
        std::cout << "Invalid index" << std::endl;
        return false;
    }
    
    if ((row > 0) && (row <= DepthSizes[IX].height) && (col > 0) && (col <= DepthSizes[IX].width)) {
        // assert (depthStorage[IX][(col - 1) * DepthSizes[IX].height + (row - 1)] == depthMaps[IX](row - 1, col - 1));
        depth = depthMaps[IX](row - 1, col - 1);
        return true;
    }
    return false;
}


bool StreamFile::StartDepthQuerySequence() {
    depthfid.open(fn, std::ios_base::binary | std::ios_base::in);
    if(!depthfid.is_open()) {
        std::cout << "Error opening " << fn << std::endl;
        return false;
    }
    return true;
}


bool StreamFile::DepthQuery(const int IX, const std::streamoff& row, const std::streamoff& col, float& depth) {
    if(IX >= NumberOfEntries()) {
        std::cout << "Invalid index" << std::endl;
        return false;
    }
    StreamFile::DInfo dInfo = DepthSizes[IX];
    if ((row > 0) && (row <= dInfo.height) && (col > 0) && (col <= dInfo.width)) {
        std::streamoff offset = streamDepthOffset[IX] + (col - 1) * dInfo.height * 4 + (row - 1) * 4;
        depthfid.seekg(offset, std::ios_base::beg);
        depthfid.read((char*)&depth, sizeof(depth));
        return true;
    }
    return false;
}


void StreamFile::DepthSizeQuery(int& height, int& width) const
{
    // TODO: currently, we assume all depth maps have the same size, may need to change it later
    height = DepthSizes[0].height;
    width = DepthSizes[0].width;
}

void StreamFile::EndDepthQuerySequence() {
    depthfid.close();
}


const Eigen::Matrix<float, Eigen::Dynamic, 4>& StreamFile::GetViewMatrix() const {
    return viewMatrix;
}


const Eigen::Matrix<float, Eigen::Dynamic, 4>& StreamFile::GetProjectionMatrix() const {
    return projMatrix;
}


const Eigen::Matrix<float, Eigen::Dynamic, 4>& StreamFile::GetTransformMatrix() const {
    return transformMatrix;
}


void StreamFile::WriteCameraMatrixBinary(const std::string& fileCameraMatrix)
{
    // storage format: [4 * #cameras, 4]
    // unsigned int rows = viewMatrix.rows(), cols = viewMatrix.cols();
    // assert (std::numeric_limits<unsigned int>::max() > rows);
    // assert (std::numeric_limits<unsigned int>::max() > cols);

    float rows = float(viewMatrix.rows()), cols = float(viewMatrix.cols());
    std::cout << std::endl << rows << " " << cols << std::endl;
   
    std::ofstream outFileStream(fileCameraMatrix, std::ios_base::binary | std::ios_base::out);
    
    if (!outFileStream.is_open()) {
        std::cout << "Unable to open binary file for saving camera matrices." << std::endl;
        exit(EXIT_FAILURE);
    }

    outFileStream.write((char*)(&rows), sizeof(float));
    outFileStream.write((char*)(&cols), sizeof(float));
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            outFileStream.write((char*)&viewMatrix(i, j), sizeof(float));
        }
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            outFileStream.write((char*)&projMatrix(i, j), sizeof(float));
        }
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            outFileStream.write((char*)&transformMatrix(i, j), sizeof(float));
        }
    }
    outFileStream.close();
}


enum StreamTypeEnum StreamFile::GetStreamType() const {
    return streamType;
}

}