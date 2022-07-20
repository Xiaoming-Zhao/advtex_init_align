#include <vector>
#include <chrono>
#include <limits>
#include <math.h>

#include <boost/gil/image.hpp>
#include <boost/gil/typedefs.hpp>
#include <boost/gil/extension/io/png.hpp>

#include <omp.h>

#include "TextureAlignment.h"
#include "Utils.h"

namespace NS_TextureAlign {

TextureAligner::TextureAligner() {}


TextureAligner::~TextureAligner() {
    ClearCues();
}


const TextureAligner::TexAlignerDataStructure& TextureAligner::GetData() const {
    return texAlignerData;
}


const Eigen::MatrixXf& TextureAligner::GetNDC() const
{
    return texAlignerData.NDC;
}


const std::vector<Eigen::MatrixXf> TextureAligner::GetValidFaceAreaMatrix()
{
    std::vector<Eigen::MatrixXf> validFaceAreaMatrix;
    for(int k = 0; k < texAlignerData.faceVertexVisibilityMatrix.size(); ++k) {
        // [#cameras, #faces]
        Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> tmpValidMatrix;
        ComputeValidMatrix(k, tmpValidMatrix);

        validFaceAreaMatrix.push_back(texAlignerData.faceAreaMatrix[k]);
        
        for (int m = 0; m < texAlignerData.faceAreaMatrix[k].rows(); m++) {
            for (int f = 0; f < texAlignerData.faceAreaMatrix[k].cols(); f++) {
                if (!tmpValidMatrix(m, f)) validFaceAreaMatrix[k](m, f) = 0.0f;
            }
        }
    }
    return validFaceAreaMatrix;
}


void TextureAligner::ClearCues() {
    texAlignerData.vertexHomoCoords.resize(0, 0);
    texAlignerData.NDC.resize(0, 0);
    // faceCenterNDC.resize(0, 0);
    std::vector<int>().swap(texAlignerData.homoCoordNDCIdx);

    std::vector<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>>().swap(texAlignerData.faceVertexVisibilityMatrix);
    std::vector<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>>().swap(texAlignerData.faceCenterPostiveZMatrix);
    std::vector<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>>().swap(texAlignerData.faceCenterVisibilityMatrix);
    std::vector<Eigen::MatrixXf>().swap(texAlignerData.faceAreaMatrix);
    std::vector<Eigen::MatrixXf>().swap(texAlignerData.faceCameraDistMatrix);
    std::vector<Eigen::MatrixXf>().swap(texAlignerData.faceVertexPerceptionConsistencyMatrix);

    std::vector<std::map<int, float>>().swap(texAlignerData.vertexDepths);

    texAlignerData.validVertexCameraPair.resize(0, 0);
    std::vector<std::map<int, std::vector<float>>>().swap(texAlignerData.vertexColors);
    std::vector<std::vector<float>>().swap(texAlignerData.meanVertexColors);
    std::vector<Eigen::MatrixXf>().swap(texAlignerData.faceColorMatrix);

    std::vector<int>().swap(texAlignerData.textureForTriangles);
}


void TextureAligner::ComputeNDC(const NS_StreamFile::StreamFile& streamInfos,
                                const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
                                // const bool& flagTopOneAssignedMtl, const std::string& fileNDC, const bool& newFile,
                                const int& globalDebugFlag, const std::string& prompt)
{
    auto start = std::chrono::high_resolution_clock::now();

    texAlignerData.vertexHomoCoords = data.Points.cast<float>();
    // [4, #vertices], homogeneous coordinates
    texAlignerData.vertexHomoCoords.conservativeResize(texAlignerData.vertexHomoCoords.rows() + 1, Eigen::NoChange);
    texAlignerData.vertexHomoCoords.row(texAlignerData.vertexHomoCoords.rows() - 1).fill(1.0f);
    // [4 * #cameras, #points]
    Eigen::MatrixXf res = streamInfos.GetTransformMatrix() * texAlignerData.vertexHomoCoords;

    for(int k = 0; k < streamInfos.NumberOfEntries(); ++k) {
        texAlignerData.homoCoordNDCIdx.push_back(4 * k);
        texAlignerData.homoCoordNDCIdx.push_back(4 * k + 1);
    }
    texAlignerData.NDC = res(texAlignerData.homoCoordNDCIdx, Eigen::all);
    for(int k = 0; k < streamInfos.NumberOfEntries(); ++k) {
        texAlignerData.NDC(2 * k, Eigen::all) = texAlignerData.NDC(2 * k, Eigen::all).cwiseQuotient(res(4 * k + 3, Eigen::all));
        texAlignerData.NDC(2 * k + 1, Eigen::all) = -texAlignerData.NDC(2 * k + 1, Eigen::all).cwiseQuotient(res(4 * k + 3, Eigen::all));
    }
    // [2 * #cameras, #points]
    texAlignerData.NDC = (texAlignerData.NDC.array() + 1.0) / 2.0;

    // if (!flagTopOneAssignedMtl) WriteNDC(fileNDC, newFile);

    if (globalDebugFlag == 1) {
        int debug_ndc_cnt = ((texAlignerData.NDC.array() >= 0.0f) && (texAlignerData.NDC.array() < 1.0f)).cast<int>().array().sum();
        float totalSum = texAlignerData.NDC.array().sum();
        std::cout << prompt + "[Debug] texAlignerData.NDC cnt " << texAlignerData.NDC.rows() << " " << texAlignerData.NDC.cols() << " " << debug_ndc_cnt << " " << totalSum << std::endl;
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << prompt + "complete computing texAlignerData.NDC in " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms" << std::endl;
}


void TextureAligner::ComputeCues(NS_StreamFile::StreamFile& streamInfos, const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
                                 const bool& flagFaceAreaNDC, const int& globalDebugFlag, const std::string& prompt)
{
    ComputeFaceVertexVisibility(streamInfos, data, globalDebugFlag, prompt);
    ComputeFaceCenterVisibility(streamInfos, data, globalDebugFlag, prompt);
    ComputeFaceCenterPositiveZ(streamInfos, data, globalDebugFlag, prompt);
    ComputeFaceArea(streamInfos, data, flagFaceAreaNDC, globalDebugFlag, prompt);
    ComputeFaceCameraDist(streamInfos, data, globalDebugFlag, prompt);
    ComputeFaceVertexPerceptionConsistency(streamInfos, data, globalDebugFlag, prompt);
}


void TextureAligner::WriteNDC(const std::string& fileNDC, const bool& newFile)
{
    unsigned int rows = texAlignerData.NDC.rows(), cols = texAlignerData.NDC.cols();
    assert (std::numeric_limits<unsigned int>::max() > rows);
    assert (std::numeric_limits<unsigned int>::max() > cols);

    // storage format: [2 * #cameras, #points]
    std::ofstream outFileStream;
    if (newFile) outFileStream.open(fileNDC, std::ios_base::binary | std::ios_base::out);
    else outFileStream.open(fileNDC, std::ios_base::binary | std::ios_base::app);
    
    if (!outFileStream.is_open()) {
        std::cout << "Unable to open binary file for saving texAlignerData.NDC." << std::endl;
        exit(EXIT_FAILURE);
    }

    outFileStream.write((char*)(&rows), sizeof(unsigned int));
    outFileStream.write((char*)(&cols), sizeof(unsigned int));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            outFileStream.write((char*)&texAlignerData.NDC(i, j), sizeof(float));
        }
    }
    outFileStream.close();
}


void TextureAligner::WriteBin(const std::string& binDir, const bool& newFile, const int& globalDebugFlag, const std::string& prompt)
{
    auto start = std::chrono::high_resolution_clock::now();

    std::ofstream outFileStream;

    bool orderC = false;

    // write hyperparameters
    if (newFile) {
        nlohmann::json infoJson;
        infoJson["c_order"] = orderC;
        infoJson["n_cams"] = int(texAlignerData.faceAreaMatrix[0].rows());
        infoJson["max_val_to_avoid_nan"] = MAX_VAL_TO_AVOID_NAN;
        infoJson["depth_discrepancy_max_penalty"] = DEPTH_DISCREPANCY_MAX_PENALTY;
    
        outFileStream.open(binDir + "/info.json", std::ios_base::out);
        outFileStream << infoJson << std::endl;
        outFileStream.close();
    }

    // texAlignerData.NDC
    // mat format [2 * #cameras, #points], bin format [#points, 2 * #cameras]
    std::string fileNDC = binDir + "/ndc.bin";
    if (newFile) outFileStream.open(fileNDC, std::ios_base::binary | std::ios_base::out);
    else outFileStream.open(fileNDC, std::ios_base::binary | std::ios_base::app);
    NS_Utils::WriteMatrixBin(texAlignerData.NDC, outFileStream, orderC);
    outFileStream.close();

    std::cout << prompt + "finish texAlignerData.NDC" << std::endl;

    // face area
    // mat format [#cameras, #facess], bin format [#faces, #cameras]
    std::string fileFaceArea = binDir + "/face_area.bin";
    if (newFile) outFileStream.open(fileFaceArea, std::ios_base::binary | std::ios_base::out);
    else outFileStream.open(fileFaceArea, std::ios_base::binary | std::ios_base::app);
    for (int k = 0; k < texAlignerData.faceAreaMatrix.size(); k++) NS_Utils::WriteMatrixBin(texAlignerData.faceAreaMatrix[k], outFileStream, orderC);
    outFileStream.close();

    std::cout << prompt + "finish face area" << std::endl;

    // face camera distance
    // mat format [#cameras, #facess], bin format [#faces, #cameras]
    std::string fileFaceCamDist = binDir + "/face_cam_dist.bin";
    if (newFile) outFileStream.open(fileFaceCamDist, std::ios_base::binary | std::ios_base::out);
    else outFileStream.open(fileFaceCamDist, std::ios_base::binary | std::ios_base::app);
    for (int k = 0; k < texAlignerData.faceCameraDistMatrix.size(); k++) NS_Utils::WriteMatrixBin(texAlignerData.faceCameraDistMatrix[k], outFileStream, orderC);
    outFileStream.close();

    std::cout << prompt + "finish face camera distance" << std::endl;

    std::string fileFaceCamDistRef = binDir + "/face_cam_dist_ref.bin";
    if (newFile) outFileStream.open(fileFaceCamDistRef, std::ios_base::binary | std::ios_base::out);
    else outFileStream.open(fileFaceCamDistRef, std::ios_base::binary | std::ios_base::app);
    for (int k = 0; k < texAlignerData.faceCameraDistRefMatrix.size(); k++) NS_Utils::WriteMatrixBin(texAlignerData.faceCameraDistRefMatrix[k], outFileStream, orderC);
    outFileStream.close();

    std::cout << prompt + "finish face camera reference distance" << std::endl;

    // face perception consistency distance
    // mat format [#cameras * 3, #facess], bin format [#faces, #cameras * 3]
    std::string fileFacePerceptionConsist = binDir + "/face_perception_consist.bin";
    if (newFile) outFileStream.open(fileFacePerceptionConsist, std::ios_base::binary | std::ios_base::out);
    else outFileStream.open(fileFacePerceptionConsist, std::ios_base::binary | std::ios_base::app);
    for (int k = 0; k < texAlignerData.faceVertexPerceptionConsistencyMatrix.size(); k++) NS_Utils::WriteMatrixBin(texAlignerData.faceVertexPerceptionConsistencyMatrix[k], outFileStream, orderC);
    outFileStream.close();

    std::cout << prompt + "finish face perception consistency" << std::endl;

    // face color per camera view
    // mat format [#cameras, #facess], bin format [#faces, #cameras]
    std::string fileFacePerception = binDir + "/face_perception.bin";
    if (newFile) outFileStream.open(fileFacePerception, std::ios_base::binary | std::ios_base::out);
    else outFileStream.open(fileFacePerception, std::ios_base::binary | std::ios_base::app);
    for (int k = 0; k < texAlignerData.faceColorMatrix.size(); k++) NS_Utils::WriteMatrixBin(texAlignerData.faceColorMatrix[k], outFileStream, orderC);
    outFileStream.close();

    std::cout << prompt + "finish face perception per camera view" << std::endl;

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << prompt + "complete saving texAlignerData.NDC + cues to binary in " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms" << std::endl;
}


void TextureAligner::ComputeFaceVertexVisibility(const NS_StreamFile::StreamFile& streamInfos, const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
        const int& globalDebugFlag, const std::string& prompt)
{
    auto start = std::chrono::high_resolution_clock::now();
    
    for(int k = 0; k < data.pointTriangles.size(); ++k) {
        texAlignerData.faceVertexVisibilityMatrix.emplace_back(Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>(streamInfos.NumberOfEntries(), data.pointTriangles[k].size()));
    }

#pragma omp parallel for
    for(int k = 0; k < data.pointTriangles.size(); ++k) {
        // Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> tmpMatrix(streamInfos.NumberOfEntries(), data.pointTriangles[k].size());
        Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& tmpMatrix = texAlignerData.faceVertexVisibilityMatrix[k];
        for(int f = 0; f < data.pointTriangles[k].size(); ++f) {
            for(int m = 0 ; m < streamInfos.NumberOfEntries(); ++m) {
                // NOTE: if later we want to check visibility with height/width, pay attention to what x-axis stands for
                Eigen::MatrixXf xcoords = texAlignerData.NDC(2 * m, data.pointTriangles[k][f].c); 
                Eigen::MatrixXf ycoords = texAlignerData.NDC(2 * m + 1, data.pointTriangles[k][f].c); 
                tmpMatrix(m,f) = (xcoords.array()>=0.0).all() && (xcoords.array()<1.0).all() && (ycoords.array()>=0.0).all() && (ycoords.array()<1.0).all();
            }
        }
        // texAlignerData.faceVertexVisibilityMatrix.push_back(tmpMatrix);
    }

    if (globalDebugFlag == 1) {
        int debug_face_vis_cnt = 0;
        for (auto it: texAlignerData.faceVertexVisibilityMatrix) debug_face_vis_cnt += it.cast<int>().array().sum();
        std::cout << prompt + "[Debug] face vis cnt: " << debug_face_vis_cnt << std::endl;

    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << prompt + "[validity] complete computing face vertex visibility in " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms" << std::endl;
}

void TextureAligner::ComputerFaceCenterNDC(Eigen::MatrixXf& faceCenterNDC, const Eigen::Vector4f& center3DCoord,
                                           const NS_StreamFile::StreamFile& streamInfos)
{
    // NOTE: TODO: if later we compute resCenter with ONLY viewMatrix,
    // then we need to multiply projMatrix here to get texAlignerData.NDC
    // [4 * #cameras, 1]
    Eigen::MatrixXf resCenter = streamInfos.GetTransformMatrix() * center3DCoord;


    // NOTE: Eigen::seq has double ends inclusive, https://eigen.tuxfamily.org/dox-devel/namespaceEigen.html#a0c04400203ca9b414e13c9c721399969
    resCenter(Eigen::seq(0, 4 * streamInfos.NumberOfEntries() - 1, 4), 0) = resCenter(Eigen::seq(0, 4 * streamInfos.NumberOfEntries() - 1, 4), 0).cwiseQuotient(resCenter(Eigen::seq(3, 4 * streamInfos.NumberOfEntries() - 1, 4), 0));
    resCenter(Eigen::seq(1, 4 * streamInfos.NumberOfEntries() - 1, 4), 0) = -resCenter(Eigen::seq(1, 4 * streamInfos.NumberOfEntries() - 1, 4), 0).cwiseQuotient(resCenter(Eigen::seq(3, 4 * streamInfos.NumberOfEntries() - 1, 4), 0));
    // [2 * #cameras, 1]
    faceCenterNDC = resCenter(texAlignerData.homoCoordNDCIdx, Eigen::all);
    faceCenterNDC = (faceCenterNDC.array() + 1.0) / 2.0;
}

void TextureAligner::ComputeFaceCenterVisibility(const NS_StreamFile::StreamFile& streamInfos, const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
        const int& globalDebugFlag, const std::string& prompt)
{
    auto start = std::chrono::high_resolution_clock::now();

    for(int k = 0; k < data.pointTriangles.size(); ++k) {
        texAlignerData.faceCenterVisibilityMatrix.emplace_back(Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>(streamInfos.NumberOfEntries(), data.pointTriangles[k].size()));
    }

#pragma omp parallel for
    for(int k = 0; k < data.pointTriangles.size(); ++k) {
        // Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> tmpMatrix(streamInfos.NumberOfEntries(), data.pointTriangles[k].size());
        Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& tmpMatrix = texAlignerData.faceCenterVisibilityMatrix[k];

        for(int f = 0; f < data.pointTriangles[k].size(); ++f) {
            Eigen::Matrix<float, 4, 3> faceCoords = texAlignerData.vertexHomoCoords(Eigen::all, data.pointTriangles[k][f].c);
            Eigen::Vector4f centerCoord = faceCoords.rowwise().mean();

            Eigen::MatrixXf faceCenterNDC;
            ComputerFaceCenterNDC(faceCenterNDC, centerCoord, streamInfos);

            Eigen::MatrixXf xcoords = faceCenterNDC(Eigen::seq(0, 2 * streamInfos.NumberOfEntries() - 1, 2), 0);
            Eigen::MatrixXf ycoords = faceCenterNDC(Eigen::seq(1, 2 * streamInfos.NumberOfEntries() - 1, 2), 0);        
            tmpMatrix(Eigen::all, f) = (xcoords.array() >= 0.0) && (xcoords.array() < 1.0) && (ycoords.array() >= 0.0) && (ycoords.array() < 1.0);

        }
    }
    
    if (globalDebugFlag == 1) {
        int debug_face_center_vis_cnt = 0;
        for (auto it: texAlignerData.faceCenterVisibilityMatrix) debug_face_center_vis_cnt += it.cast<int>().array().sum();
        std::cout << prompt + "[Debug] face center vis cnt: " << debug_face_center_vis_cnt << std::endl;
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << prompt + "[validity] complete computing face center visibility in " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms" << std::endl;
}

void TextureAligner::ComputeFaceCenterPositiveZ(const NS_StreamFile::StreamFile& streamInfos, const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
        const int& globalDebugFlag, const std::string& prompt)
{
    auto start = std::chrono::high_resolution_clock::now();

    for(int k = 0; k < data.pointTriangles.size(); ++k) {
        texAlignerData.faceCenterPostiveZMatrix.emplace_back(Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>(streamInfos.NumberOfEntries(), data.pointTriangles[k].size()));
    }

#pragma omp parallel for
    for(int k = 0; k < data.pointTriangles.size(); ++k) {
        // Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> tmpMatrix(streamInfos.NumberOfEntries(), data.pointTriangles[k].size());
        Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& tmpMatrix = texAlignerData.faceCenterPostiveZMatrix[k];

        for(int f = 0; f < data.pointTriangles[k].size(); ++f) {
            Eigen::MatrixXf faceCoords = texAlignerData.vertexHomoCoords(Eigen::all, data.pointTriangles[k][f].c);
            Eigen::MatrixXf centerCoord = faceCoords.rowwise().mean();
            Eigen::MatrixXf res = streamInfos.GetTransformMatrix() * centerCoord;
            tmpMatrix(Eigen::all, f) = res(Eigen::seq(2, 4 * streamInfos.NumberOfEntries() - 1, 4), 0).array() >= 0.01;
        }
        // texAlignerData.faceCenterPostiveZMatrix.push_back(tmpMatrix);
    }

    if (globalDebugFlag == 1) {
        int debug_pos_z_cnt = 0;
        for (auto it: texAlignerData.faceCenterPostiveZMatrix) debug_pos_z_cnt += it.cast<int>().array().sum();
        std::cout << prompt + "[Debug] positive z cnt: " << debug_pos_z_cnt << std::endl;
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << prompt + "[validity] complete computing face positive z in " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms" << std::endl;
}

void TextureAligner::ComputeFaceArea(const NS_StreamFile::StreamFile& streamInfos, const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
                                     const bool& flagFaceAreaNDC, const int& globalDebugFlag, const std::string& prompt)
{
    auto start = std::chrono::high_resolution_clock::now();

    for(int k = 0; k < data.pointTriangles.size(); ++k) {
        texAlignerData.faceAreaMatrix.emplace_back(Eigen::MatrixXf::Zero(streamInfos.NumberOfEntries(), data.pointTriangles[k].size()));
    }

    for(int k = 0; k < data.pointTriangles.size(); ++k) {
        // Eigen::MatrixXf tmpMatrix(streamInfos.NumberOfEntries(), data.pointTriangles[k].size());
        Eigen::MatrixXf& tmpMatrix = texAlignerData.faceAreaMatrix[k];

#pragma omp parallel for
        for(int f = 0; f < data.pointTriangles[k].size(); ++f) {
            float rgbHeight, rgbWidth;
            float maxArea = 0.0f;
            for(int m = 0; m < streamInfos.NumberOfEntries(); ++m) {
                Eigen::MatrixXf xcoords = texAlignerData.NDC(2 * m, data.pointTriangles[k][f].c);  
                Eigen::MatrixXf ycoords = texAlignerData.NDC(2 * m + 1, data.pointTriangles[k][f].c);
                if (!flagFaceAreaNDC) {
                    streamInfos.RGBSizeQuery(m, rgbHeight, rgbWidth);
                    xcoords *= rgbHeight;
                    ycoords *= rgbWidth;
                }
                Eigen::Vector2f dir1 = {xcoords(0, 1) - xcoords(0,0), ycoords(0, 1) - ycoords(0, 0)};
                Eigen::Vector2f dir2 = {xcoords(0, 2) - xcoords(0,0), ycoords(0, 2) - ycoords(0, 0)};
                if (!texAlignerData.faceVertexVisibilityMatrix[k](m, f)) {
                    tmpMatrix(m, f) = 0.0;
                } else{
                    // has to be set to zero if texAlignerData.faceVertexVisibilityMatrix is false;
                    tmpMatrix(m, f) = std::fabs(dir1(0) * dir2(1) - dir1(1) * dir2(0)) / 2.0f;
                }

                if (tmpMatrix(m, f) > maxArea) maxArea = tmpMatrix(m, f);
            }
        }
    }

    if (globalDebugFlag == 1) {
        float debugFaceAreaSum = 0.0f;
        for (auto it: texAlignerData.faceAreaMatrix) debugFaceAreaSum += it.array().sum();
        std::cout << prompt + "[Debug] face area sum: " << debugFaceAreaSum << std::endl;

        float maxVal = 0.0f;
        for (auto it: texAlignerData.faceAreaMatrix) {
            if (maxVal < it.maxCoeff()) maxVal = it.maxCoeff();
        }
        std::cout << prompt + "[Debug] face area matrix max val: " << maxVal << std::endl;
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << prompt + "[score] complete computing face area in " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms" << std::endl;
}

void TextureAligner::ComputeFaceCameraDist(const NS_StreamFile::StreamFile& streamInfos, const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
        const int& globalDebugFlag, const std::string& prompt)
{

    auto start = std::chrono::high_resolution_clock::now();

    const int nCameras = streamInfos.NumberOfEntries();
    float distDiffAbsTol = 0.25;

    texAlignerData.vertexDepths.resize(data.Points.cols(), std::map<int, float>());

    int depth_height, depth_width;
    streamInfos.DepthSizeQuery(depth_height, depth_width);

    Eigen::MatrixXf matVertexDepth = Eigen::MatrixXf::Zero(data.Points.cols(), nCameras);

    for (int k = 0; k < data.pointTriangles.size(); k++) {
        // [#cameras, #faces]
        Eigen::MatrixXf discrepancyMatrix(nCameras, data.pointTriangles[k].size());
        Eigen::MatrixXf distRefMatrix(nCameras, data.pointTriangles[k].size());
        // Eigen::MatrixXf& discrepancyMatrix = texAlignerData.faceCameraDistMatrix[k];

#pragma omp parallel for
        for(int f = 0; f < data.pointTriangles[k].size(); ++f) {
            // get 3D coordinates of face vertices and center.
            Eigen::Matrix<float, 4, 3> faceCoords = texAlignerData.vertexHomoCoords(Eigen::all, data.pointTriangles[k][f].c);
            Eigen::Vector4f centerCoord = faceCoords.rowwise().mean();

            // get 3D coordinates in each camera's coordinate system
            // [4 * #cameras, 3], the last 3 is for three vertices
            Eigen::MatrixXf resVertices = streamInfos.GetViewMatrix() * faceCoords;
            // [4 * #cameras, 1]
            Eigen::MatrixXf resCenter = streamInfos.GetViewMatrix() * centerCoord;

            // get distance directly from 3D coordinates
            Eigen::Matrix<float, Eigen::Dynamic, 4> distNoDepth(nCameras, 4);
            for (int i = 0; i < nCameras; i++) {
                distNoDepth.block<1, 3>(i, 0) = resVertices(Eigen::seq(4 * i, 4 * i + 2, 1), Eigen::all).colwise().norm();
                distNoDepth(i, 3) = resCenter(Eigen::seq(4 * i, 4 * i + 2, 1), Eigen::all).norm();
            }

            // get texAlignerData.NDC coordinates, [2 * #cameras, 3], 2 is for XY
            Eigen::Matrix<float, Eigen::Dynamic, 3> faceNDC(2 * nCameras, 3);
            faceNDC = texAlignerData.NDC(Eigen::all, data.pointTriangles[k][f].c);

            Eigen::MatrixXf faceCenterNDC;
            ComputerFaceCenterNDC(faceCenterNDC, centerCoord, streamInfos);

            // get pixel coordinates
            Eigen::Matrix<float, Eigen::Dynamic, 3> facePixelCoords(2 * nCameras, 3);
            facePixelCoords(Eigen::seq(0, 2 * nCameras - 1, 2), Eigen::all) = faceNDC(Eigen::seq(0, 2 * nCameras - 1, 2), Eigen::all) * depth_height;
            facePixelCoords(Eigen::seq(1, 2 * nCameras - 1, 2), Eigen::all) = faceNDC(Eigen::seq(1, 2 * nCameras - 1, 2), Eigen::all) * depth_width;
            facePixelCoords = facePixelCoords.array().round();

            Eigen::Matrix<float, Eigen::Dynamic, 1> centerPixelCoords(2 * nCameras, 1);
            centerPixelCoords(Eigen::seq(0, 2 * nCameras - 1, 2), Eigen::all) = faceCenterNDC(Eigen::seq(0, 2 * nCameras - 1, 2), Eigen::all) * depth_height;
            centerPixelCoords(Eigen::seq(1, 2 * nCameras - 1, 2), Eigen::all) = faceCenterNDC(Eigen::seq(1, 2 * nCameras - 1, 2), Eigen::all) * depth_width;
            centerPixelCoords = centerPixelCoords.array().round();

            Eigen::Matrix<float, Eigen::Dynamic, 4> distWithDepth = Eigen::Matrix<float, Eigen::Dynamic, 4>::Zero(nCameras, 4);
            for(int m = 0; m < nCameras; ++m) {
                std::streamoff row, col, offset;
                float depth = 0.0;

                for (int v_i = 0; v_i < 4; v_i++) {
                    if (v_i < 3) {
                        // NOTE: not sure whether this cast is safe
                        row = facePixelCoords(2 * m, v_i);
                        col = facePixelCoords(2 * m + 1, v_i);
                    } else {
                        // processing face center
                        row = centerPixelCoords(2 * m, 0);
                        col = centerPixelCoords(2 * m + 1, 0);
                    }

                    // NOTE: here I get pixel coordinates via reversely mimicing matlab's code rot90(fliplr(depth{camera}))
                    // Weird, seems like no need to reversely mimic fliplr in matlab
                    // Reason is not clear at this time
                    // col = NS_StreamFile::DEPTH_WIDTH - col;

                    if (streamInfos.DepthQueryFromMem(m, row, col, depth)) {

                        if (depth < 1e-3) {
                            // This branch approximates "mask". If depth equals zero, we treat it as no content here. 
                            distWithDepth(m, v_i) = std::numeric_limits<float>::max();
                        } else {
                            if (v_i < 3) {
                                // store depth for future usage
                                // texAlignerData.vertexDepths[data.pointTriangles[k][f].c[v_i]].insert(std::make_pair(m, depth));
                                matVertexDepth(data.pointTriangles[k][f].c[v_i], m) = depth;
    
                                // Pay attention to Eigen:seq, we only need two elements (x, y) here
                                distWithDepth(m, v_i) = std::sqrt(resVertices(Eigen::seq(4 * m, 4 * m + 1, 1), v_i).colwise().squaredNorm().value() + std::pow(depth, 2.0));
                            } else {
                                distWithDepth(m, v_i) = std::sqrt(resCenter(Eigen::seq(4 * m, 4 * m + 1, 1), 0).colwise().squaredNorm().value() + std::pow(depth, 2.0));
                            }
                        }
                    } else {
                        distWithDepth(m, v_i) = std::numeric_limits<float>::max();
                    } 
                }
            }

            Eigen::Matrix<float, Eigen::Dynamic, 4> distDiff = distWithDepth - distNoDepth;
            distDiff = distDiff.cwiseAbs();
            discrepancyMatrix(Eigen::all, f) = distDiff.rowwise().maxCoeff();
            // discrepancyMatrix(Eigen::all, f) = (distDiff.array() < distDiffAbsTol).rowwise().all();

            for (int m = 0; m < nCameras; m++) {
                for (int i = 0; i < 4; i++) {
                    if (discrepancyMatrix(m, f) == distDiff(m, i)) {
                        distRefMatrix(m, f) = std::max(distWithDepth(m, i), distNoDepth(m, i));
                        break;
                    }
                }
            }
        }

#pragma omp parallel for
        for (int i = 0; i < discrepancyMatrix.rows(); i++) {
            for (int j = 0; j <  discrepancyMatrix.cols(); j++) {
                if (streamInfos.GetStreamType() == NS_StreamFile::StreamTypeEnum::STREAM_APPLE) {
                    // NOTE: this is not suitable since discrepancy will be larger at far distance,
                    // absolute tolerance is not adaptive.
                    if (discrepancyMatrix(i, j) > 0.25) {
                        discrepancyMatrix(i, j) = 5;
                    }
                } else {

                    // NOTE: this is mainly for the Fountain scene in 
                    // Q.-Y. Zhou and V. Koltun,
                    // Color Map Optimization for 3D Reconstruction with Consumer Depth Cameras,
                    // SIGGRAPH 2014
                    // Ref: https://github.com/isl-org/Open3D/blob/70326b8/cpp/pybind/pipelines/color_map/color_map.cpp#L92
                    if (discrepancyMatrix(i, j) > 0.25) {
                        discrepancyMatrix(i, j) = DEPTH_DISCREPANCY_MAX_PENALTY;
                    }
                    // std::cout << "here" << std::endl;
                }
            }
        }

        texAlignerData.faceCameraDistMatrix.push_back(discrepancyMatrix);
        texAlignerData.faceCameraDistRefMatrix.push_back(distRefMatrix);
    }

    for (int p = 0; p < data.Points.cols(); p++) {
        for (int m = 0; m < nCameras; m++) {
            if (matVertexDepth(p, m) != 0) texAlignerData.vertexDepths[p].insert(std::make_pair(m, matVertexDepth(p, m)));
        }
    }

    // streamInfos.EndDepthQuerySequence();

    if (globalDebugFlag == 1) {
        float debugFaceCamDistSum = 0.0f;
        for (auto it: texAlignerData.faceCameraDistMatrix) debugFaceCamDistSum += it.array().sum();
        std::cout << prompt + "[Debug] face camera distance sum: " << debugFaceCamDistSum << std::endl;

        float minVal = 0.0f, maxVal = 0.0f;
        float minReasonableVal = 0.0f;
        for (auto it: texAlignerData.faceCameraDistMatrix) {
            if (minVal > it.minCoeff()) minVal = it.minCoeff();
            if (maxVal < it.maxCoeff()) maxVal = it.maxCoeff();
            for (int i = 0; i < it.rows(); i++) {
                for (int j = 0; j < it.cols(); j++) {
                    if ((it(i, j) > -APPLE_MAX_DEPTH) && (minReasonableVal > it(i, j))) minReasonableVal = it(i, j);
                }
            }
        }
        std::cout << prompt + "[Debug] face camera distance matrix val range: [" << minVal << ", " << maxVal \
                  << "], min reasonable val: " << minReasonableVal << std::endl;
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << prompt + "[score] complete computing face camera distance in " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms" << std::endl;
}


void TextureAligner::ComputeFaceVertexPerceptionConsistency(NS_StreamFile::StreamFile& streamInfos,
                            const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
                            const int& globalDebugFlag, const std::string& prompt)
{
    auto start = std::chrono::high_resolution_clock::now();

    RetrieveVertexColor(streamInfos, data, globalDebugFlag, prompt);

    for(int k = 0; k < data.pointTriangles.size(); ++k) {
        texAlignerData.faceVertexPerceptionConsistencyMatrix.emplace_back(Eigen::MatrixXf::Zero(streamInfos.NumberOfEntries(), data.pointTriangles[k].size()));
    }

    for(int k = 0; k < data.pointTriangles.size(); ++k) {
        // Eigen::MatrixXf tmpMatrix(streamInfos.NumberOfEntries(), data.pointTriangles[k].size());
        Eigen::MatrixXf& tmpMatrix = texAlignerData.faceVertexPerceptionConsistencyMatrix[k];

        // [#cameras, #faces]
        Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> tmpValidMatrix;
        ComputeValidMatrix(k, tmpValidMatrix);

        // [#cameras, #faces]
        Eigen::MatrixXf tmpFaceColorMatrix = Eigen::MatrixXf::Constant(streamInfos.NumberOfEntries() * 3, data.pointTriangles[k].size(), -1.0);

#pragma omp parallel for
        for(int f = 0; f < data.pointTriangles[k].size(); ++f) {
            float maxDiff = 0.0f;

            for(int viewIdx = 0; viewIdx < streamInfos.NumberOfEntries(); ++viewIdx) {
                if (tmpValidMatrix(viewIdx, f))
                {
                    float meanFaceR = 0.0f, meanFaceG = 0.0f, meanFaceB = 0.0f;
                    char flagNaN = 0;
                    // check whether mean color value of vertex is NaN
                    for (int v = 0; v < data.pointTriangles[k][f].length; v++) {
                        if (std::isnan(texAlignerData.meanVertexColors[data.pointTriangles[k][f].c[v]][0]) ||
                            std::isnan(texAlignerData.meanVertexColors[data.pointTriangles[k][f].c[v]][1]) ||
                            std::isnan(texAlignerData.meanVertexColors[data.pointTriangles[k][f].c[v]][2]))
                        {
                            flagNaN = 1;
                            break;
                        }
                    }

                    if (flagNaN == 1) {
                        // tmpMatrix(viewIdx, f) = std::numeric_limits<float>::max();
                        tmpMatrix(viewIdx, f) = MAX_VAL_TO_AVOID_NAN;
                    } else {
                        // Face's reference average color intensity from all view where vertices are visible
                        float meanFaceRefR = (texAlignerData.meanVertexColors[data.pointTriangles[k][f].c[0]][0] + \
                                              texAlignerData.meanVertexColors[data.pointTriangles[k][f].c[1]][0] + \
                                              texAlignerData.meanVertexColors[data.pointTriangles[k][f].c[2]][0]) / 3;
                        float meanFaceRefG = (texAlignerData.meanVertexColors[data.pointTriangles[k][f].c[0]][1] + \
                                              texAlignerData.meanVertexColors[data.pointTriangles[k][f].c[1]][1] + \
                                              texAlignerData.meanVertexColors[data.pointTriangles[k][f].c[2]][1]) / 3; 
                        float meanFaceRefB = (texAlignerData.meanVertexColors[data.pointTriangles[k][f].c[0]][2] + \
                                              texAlignerData.meanVertexColors[data.pointTriangles[k][f].c[1]][2] + \
                                              texAlignerData.meanVertexColors[data.pointTriangles[k][f].c[2]][2]) / 3;
                        // Faces's color intensity from current view
                        float faceCurR = (texAlignerData.vertexColors[data.pointTriangles[k][f].c[0]][viewIdx][0] + \
                                          texAlignerData.vertexColors[data.pointTriangles[k][f].c[1]][viewIdx][0] + \
                                          texAlignerData.vertexColors[data.pointTriangles[k][f].c[2]][viewIdx][0]) / 3;
                        float faceCurG = (texAlignerData.vertexColors[data.pointTriangles[k][f].c[0]][viewIdx][1] + \
                                          texAlignerData.vertexColors[data.pointTriangles[k][f].c[1]][viewIdx][1] + \
                                          texAlignerData.vertexColors[data.pointTriangles[k][f].c[2]][viewIdx][1]) / 3;
                        float faceCurB = (texAlignerData.vertexColors[data.pointTriangles[k][f].c[0]][viewIdx][2] + \
                                          texAlignerData.vertexColors[data.pointTriangles[k][f].c[1]][viewIdx][2] + \
                                          texAlignerData.vertexColors[data.pointTriangles[k][f].c[2]][viewIdx][2]) / 3;
                        
                        // set face's color
                        tmpFaceColorMatrix(3 * viewIdx, f) = faceCurR;
                        tmpFaceColorMatrix(3 * viewIdx + 1, f) = faceCurG;
                        tmpFaceColorMatrix(3 * viewIdx + 2, f) = faceCurB;

                        // compute difference
                        float perceptDiff = sqrt(
                            std::pow(meanFaceRefR - faceCurR, 2) + std::pow(meanFaceRefG - faceCurG, 2) + std::pow(meanFaceRefB - faceCurB, 2)
                        );
                        tmpMatrix(viewIdx, f) = perceptDiff;
                        
                        if (maxDiff < perceptDiff) maxDiff = perceptDiff;
                    }

                }

            }

        }

        texAlignerData.faceColorMatrix.push_back(tmpFaceColorMatrix);
    }

    if (globalDebugFlag == 1) {
        float debug_face_vertex_color_consist_sum = 0.0f;
        for (auto it: texAlignerData.faceVertexPerceptionConsistencyMatrix) debug_face_vertex_color_consist_sum += it.array().sum();
        std::cout << prompt + "[Debug] face vertex color consistency sum: " << debug_face_vertex_color_consist_sum << std::endl;

        float minVal = 0.0f, maxVal = 0.0f;
        float minReasonableVal = 0.0f;
        for (auto it: texAlignerData.faceVertexPerceptionConsistencyMatrix) {
            if (minVal > it.minCoeff()) minVal = it.minCoeff();
            if (maxVal < it.maxCoeff()) maxVal = it.maxCoeff();
            for (int i = 0; i < it.rows(); i++) {
                for (int j = 0; j < it.cols(); j++) {
                    if ((it(i, j) > -100.0f) && (minReasonableVal > it(i, j))) minReasonableVal = it(i, j);
                }
            }
        }
        std::cout << prompt + "[Debug] face vertex color consistency val range: [" << minVal << ", " << maxVal \
                  << "], min reasonable val: " << minReasonableVal << std::endl;
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << prompt + "[score] complete computing perception consistency cue in " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms" << std::endl;
}


void TextureAligner::RetrieveVertexColor(NS_StreamFile::StreamFile& streamInfos,
                            const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
                            const int& globalDebugFlag, const std::string& prompt)
{
    std::cout << prompt + "start retrieving vertex colors ..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    const int nCameras = streamInfos.NumberOfEntries();
    const int nPoints = data.Points.cols();

    texAlignerData.validVertexCameraPair.resize(nCameras, nPoints);
    texAlignerData.validVertexCameraPair.fill(false);

    // every four elements represent a tuple of (cameraID, R, G, B)
    texAlignerData.vertexColors.resize(nPoints, std::map<int, std::vector<float>>());

    std::vector<float> tmpVec{0.0f, 0.0f, 0.0f};

    // get validity flag for (vertex, camera) pair
    for (int k = 0; k < data.pointTriangles.size(); k++) {
        // [#cameras, #faces]
        Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> tmpValidMatrix;
        ComputeValidMatrix(k, tmpValidMatrix);

        for (int f = 0; f < data.pointTriangles[k].size(); f++)  {
            for (int viewIdx = 0; viewIdx < nCameras; viewIdx++) {
                if (tmpValidMatrix(viewIdx, f)) {
                    if (!texAlignerData.validVertexCameraPair(viewIdx, data.pointTriangles[k][f].c[0])) {
                        texAlignerData.validVertexCameraPair(viewIdx, data.pointTriangles[k][f].c[0]) = true;
                        texAlignerData.vertexColors[data.pointTriangles[k][f].c[0]].insert(std::make_pair(viewIdx, tmpVec));
                    }
                    if (!texAlignerData.validVertexCameraPair(viewIdx, data.pointTriangles[k][f].c[1])) {
                        texAlignerData.validVertexCameraPair(viewIdx, data.pointTriangles[k][f].c[1]) = true;
                        texAlignerData.vertexColors[data.pointTriangles[k][f].c[1]].insert(std::make_pair(viewIdx, tmpVec));
                    }
                    if (!texAlignerData.validVertexCameraPair(viewIdx, data.pointTriangles[k][f].c[2])) {
                        texAlignerData.validVertexCameraPair(viewIdx, data.pointTriangles[k][f].c[2]) = true;
                        texAlignerData.vertexColors[data.pointTriangles[k][f].c[2]].insert(std::make_pair(viewIdx, tmpVec));
                    }
                }
            }
        }
    }

    // retrieve pixel values
#pragma omp parallel for
    for (int viewIdx = 0; viewIdx < nCameras; viewIdx++) {
        const boost::gil::rgb8_image_t* const rgbImg = streamInfos.GetRGBImage(viewIdx);
        auto rgbView = boost::gil::const_view(*rgbImg);

        for (int v = 0; v < nPoints; v++) {
            if (texAlignerData.validVertexCameraPair(viewIdx, v)) {
                // texAlignerData.NDC: [2 * #cameras, #points]
                Eigen::Array2f imgCoords = texAlignerData.NDC(Eigen::seq(2 * viewIdx, 2 * viewIdx + 1, 1), v);
                // NOTE: Important !!! Must use realheight here
                imgCoords(0) = (int)std::floor(imgCoords(0) * rgbImg->height());
                imgCoords(1) = (int)std::floor(imgCoords(1) * rgbImg->width());

                assert (imgCoords(0) >= 0 && imgCoords(0) < rgbImg->height());
                assert (imgCoords(1) >= 0 && imgCoords(1) < rgbImg->width());

                boost::gil::rgb8_pixel_t tmpPixel = rgbView(imgCoords(1), imgCoords(0));

                texAlignerData.vertexColors[v][viewIdx][0] = (float)boost::gil::get_color(tmpPixel, boost::gil::red_t()) / 255.0f;
                texAlignerData.vertexColors[v][viewIdx][1] = (float)boost::gil::get_color(tmpPixel, boost::gil::green_t()) / 255.0f;
                texAlignerData.vertexColors[v][viewIdx][2] = (float)boost::gil::get_color(tmpPixel, boost::gil::blue_t()) / 255.0f;
            }
        }
    }

    // Compute mean of vertex channel values
    // vertexColorTupleLen = 4;
    texAlignerData.meanVertexColors.resize(nPoints, std::vector<float>());

#pragma omp parallel for
    for (int v = 0; v < nPoints; v++) {

        float R = 0.0f, G = 0.0f, B = 0.0f;

        for (const auto &it: texAlignerData.vertexColors[v]) {
            R += it.second[0];
            G += it.second[1];
            B += it.second[2];
        }
        texAlignerData.meanVertexColors[v].push_back(R / texAlignerData.vertexColors[v].size());
        texAlignerData.meanVertexColors[v].push_back(G / texAlignerData.vertexColors[v].size());
        texAlignerData.meanVertexColors[v].push_back(B / texAlignerData.vertexColors[v].size());

    }

    int minValidCameras = std::numeric_limits<int>::max();
    for (int v = 0; v < nPoints; v++) {
        if (texAlignerData.vertexColors[v].size() < minValidCameras) minValidCameras = texAlignerData.vertexColors[v].size();
    }

    if (globalDebugFlag == 1) {
        std::cout << prompt + "[Debug] retrieve vertex color: minValidCameras " << minValidCameras << std::endl;
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << prompt + "... complete retrieving vertex colors in " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms" << std::endl;
}


void TextureAligner::ComputeValidMatrix(const int& modelIdx, Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& validMatrix)
{
    // [#cameras, #faces]
    validMatrix = texAlignerData.faceVertexVisibilityMatrix[modelIdx] && texAlignerData.faceCenterPostiveZMatrix[modelIdx] && texAlignerData.faceCenterVisibilityMatrix[modelIdx];
}

void TextureAligner::ComputeScoreMatrix(const int& modelIdx, const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& validMatrix, Eigen::MatrixXf& scoreMatrix,
                                        const float& unaryPotentialDummy, const float& penaltyFaceArea,
                                        const float& penaltyFaceCamDist, const float& penaltyFaceVertexPerceptionConsist)
{
    // [#cameras, #faces]
    scoreMatrix = texAlignerData.faceAreaMatrix[modelIdx] * penaltyFaceArea + \
                  texAlignerData.faceCameraDistMatrix[modelIdx] * penaltyFaceCamDist + \
                  texAlignerData.faceVertexPerceptionConsistencyMatrix[modelIdx] * penaltyFaceVertexPerceptionConsist;
    for (int viewIdx = 0; viewIdx < validMatrix.rows(); viewIdx++) {
        for (int faceIdx = 0; faceIdx < validMatrix.cols(); faceIdx++) {
            if (!validMatrix(viewIdx, faceIdx)) {
                // scoreMatrix(viewIdx, faceIdx) = -std::numeric_limits<float>::max();

                // NOTE: we only need this value is inferior to dummy so that dummy will be selectd.
                scoreMatrix(viewIdx, faceIdx) = 2 * unaryPotentialDummy;
            }
        }
    }
}

void TextureAligner::AlignTextureToMesh(const int& approachType,
                                        const float& unaryPotentialDummy, const float& penaltyFaceArea,
                                        const float& penaltyFaceCamDist, const float& penaltyFaceVertexPerceptionConsist,
                                        NS_MPSolver::MPSolver<float, int>& MP, const int& nItersMP,
                                        const float& pairwisePotentialMP,
                                        const float& pairwisePotentialOffDiagScaleDepth, const float& pairwisePotentialOffDiagScalePercept,
                                        const bool& flagSaveFullBeliefs, const std::string& fileAlBeliefs,
                                        const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
                                        const bool& flagTopOneAssignedMtl, const std::string& fileFaceCamPairs, const bool& newFile,
                                        const int& globalDebugFlag, const std::string& prompt)
{
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>> validMatrix;
    std::vector<Eigen::MatrixXf> scoreMatrix;

    for (int k = 0; k < data.pointTriangles.size(); k++) {
        // [#cameras, #faces]
        Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> tmpValidMatrix;
        ComputeValidMatrix(k, tmpValidMatrix);
        validMatrix.push_back(tmpValidMatrix);

        Eigen::MatrixXf tmpScoreMatrix;
        ComputeScoreMatrix(k, tmpValidMatrix, tmpScoreMatrix, unaryPotentialDummy, penaltyFaceArea, penaltyFaceCamDist, penaltyFaceVertexPerceptionConsist);

        // add a dummy texture
        tmpScoreMatrix.conservativeResize(tmpScoreMatrix.rows() + 1, Eigen::NoChange);
        tmpScoreMatrix.row(tmpScoreMatrix.rows() - 1).fill(unaryPotentialDummy);
        scoreMatrix.push_back(tmpScoreMatrix);

        // NOTE: when mimicing matlab pipeline, we do not have data.textureTriangles
        std::vector<int> tmpTextureForTriangles(data.pointTriangles[k].size(), 0);

        if (approachType == TextureAlignmentApproach::ArgMax) {
            AlignTextureToMeshArgMax(tmpScoreMatrix, tmpTextureForTriangles);
        } else if (approachType == TextureAlignmentApproach::MessagePassing) {
            std::string tmp_prompt = prompt + "[" + std::to_string(k) + "/" + std::to_string(data.pointTriangles.size()) + "] ";
            AlignTextureToMeshMP(k, tmpValidMatrix, tmpScoreMatrix, tmpTextureForTriangles,
                                 MP, nItersMP, pairwisePotentialMP, pairwisePotentialOffDiagScaleDepth, pairwisePotentialOffDiagScalePercept,
                                 flagSaveFullBeliefs, fileAlBeliefs, data, globalDebugFlag, tmp_prompt);
        }

        texAlignerData.textureForTriangles.insert(texAlignerData.textureForTriangles.end(), tmpTextureForTriangles.begin(), tmpTextureForTriangles.end());
    }

    if (!flagTopOneAssignedMtl) {
        GenRankedFaceCamPairs(validMatrix, scoreMatrix, fileFaceCamPairs, newFile, data, globalDebugFlag, prompt);
    }

    // assert (texAlignerData.textureForTriangles.size() == data.nTextureTriangles);
    assert (texAlignerData.textureForTriangles.size() == data.nPointTriangles);

    if (globalDebugFlag == 1) {
        int debugCueValidCnt = 0;
        for (auto it: validMatrix) debugCueValidCnt += it.cast<int>().array().sum();
        std::cout << prompt + "[Debug] Cue matrix valid cnt: " << debugCueValidCnt << std::endl;
    
        float debugScoreSum = 0.0f;
        for (auto it: scoreMatrix) debugScoreSum += it.array().sum();
        std::cout << prompt + "[Debug] Cue matrix score sum: " << debugScoreSum << std::endl;

    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << prompt + "complete texture alignment in " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms" << std::endl;
}


void TextureAligner::AlignTextureToMeshArgMax(const Eigen::MatrixXf& scoreMatrix, std::vector<int>& textureForTriangles)
{
    // [1, #faces]
    Eigen::MatrixXf faceMaxScores = scoreMatrix.colwise().maxCoeff();
    
    // NOTE: eigen seems like does not provide argmax function, need to double check later
    //slightly better way to avoid double loop: https://stackoverflow.com/questions/11430588/find-rowwise-maxcoeff-and-index-of-maxcoeff-in-eigen
    for (int viewIdx = 0; viewIdx < scoreMatrix.rows(); viewIdx++) {
        for (int faceIdx = 0; faceIdx < scoreMatrix.cols(); faceIdx++) {
            if (scoreMatrix(viewIdx, faceIdx) == faceMaxScores(0, faceIdx)) textureForTriangles[faceIdx] = viewIdx;
        }
    }
}

void TextureAligner::AlignTextureToMeshMP(const int& modelIdx,
                                          const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& validMatrix,
                                          const Eigen::MatrixXf& scoreMatrix,
                                          std::vector<int>& textureForTriangles,
                                          NS_MPSolver::MPSolver<float, int>& MP, const int& nItersMP,
                                          const float& potentialPairVal,
                                          const float& potentialPairOffDiagScaleDepth, const float& potentialPairOffDiagScalePercept,
                                          const bool& flagSaveFullBeliefs, const std::string& fileAlBeliefs,
                                          const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
                                          const int& globalDebugFlag, const std::string& prompt)
{
    auto start = std::chrono::high_resolution_clock::now();

    int nFaces = data.textureTriangles[modelIdx].size();
    int nViews = scoreMatrix.rows();    // nViews includes a dummy view
    MP.CreateGraph(nFaces, nViews);

    std::vector<float> infoUnary(nFaces * nViews, 0.0);
    for (int f = 0; f < nFaces; f++) {
        for (int v = 0; v < nViews; v++) {
            infoUnary[f * nViews + v] = scoreMatrix(v, f);
        }
    }
    std::vector<float> tmpInfoPair;
    std::vector<float> tmpPotentialsPair;

    MP.UpdateGraph(MP.GraphUpdateTypes::OnlyUnary, infoUnary, tmpInfoPair, tmpPotentialsPair);

    std::vector<float> tmpInfoUnary;
    int tmpAdjCnt = 0;

    std::vector<std::vector<float>*> allPairPotentials;
    std::map<std::pair<int, int>, int> pairIdxDict;

    for (int f = 0; f < data.infoMeshTriangleAdjacency3D[modelIdx].size(); f++) {
        for (auto it: data.infoMeshTriangleAdjacency3D[modelIdx][f]) {
            if (data.triangleGlobalIdx[modelIdx][f] < data.triangleGlobalIdx[modelIdx][it]) {
                pairIdxDict[std::make_pair(data.triangleGlobalIdx[modelIdx][f], data.triangleGlobalIdx[modelIdx][it])] = tmpAdjCnt;
                tmpAdjCnt++;
            }
        }
    }

    std::vector<std::vector<float>> allInfoPairAdjs(tmpAdjCnt);
    allPairPotentials.resize(tmpAdjCnt);


#pragma omp parallel for schedule(dynamic, 10)
    for (int f = 0; f < data.infoMeshTriangleAdjacency3D[modelIdx].size(); f++) {
        for (auto it: data.infoMeshTriangleAdjacency3D[modelIdx][f]) {
            if (data.triangleGlobalIdx[modelIdx][f] < data.triangleGlobalIdx[modelIdx][it]) {

                int tmpIdx = pairIdxDict[std::make_pair(data.triangleGlobalIdx[modelIdx][f], data.triangleGlobalIdx[modelIdx][it])];

                // we only add upper triangular information to avoid duplication.
                // Since we conduct MP on each model component separately, we use local index instead of global one
                std::vector<float> infoPairAdj{(float)f, (float)it, 1.0};
                allInfoPairAdjs[tmpIdx] = infoPairAdj;

                // get pairwise potential
                // std::vector<float> potentialPairVec(nViews * nViews, 0.0);
                std::vector<float>* potentialPairVec = new std::vector<float>(nViews * nViews, 0.0);
                // allPairPotentials.push_back(potentialPairVec);
                allPairPotentials[tmpIdx] = potentialPairVec;

                CreatePairwisePotential(modelIdx, f, it, nViews, potentialPairVal, potentialPairOffDiagScaleDepth, potentialPairOffDiagScalePercept,
                                        *potentialPairVec, validMatrix, data);

            }
        }
    }
   
    auto finishMemAllocate = std::chrono::high_resolution_clock::now();
    std::cout << prompt + "complete creating pairwise matrix in " << std::chrono::duration_cast<std::chrono::milliseconds>(finishMemAllocate - start).count() << " ms" << std::endl;
    

    for (int i = 0; i < allInfoPairAdjs.size(); i++) {
        MP.UpdateGraph(MP.GraphUpdateTypes::OnlyPair, tmpInfoUnary, allInfoPairAdjs[i], *allPairPotentials[i]);
    }

    if (globalDebugFlag == 1) {
        std::cout << prompt + "[Debug] #graphAdjInfo: " << tmpAdjCnt << std::endl;
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << prompt + "complete creating MP graph in " << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms" << std::endl;

    // check all regions/connections are added
    int nExpextedRegions = nFaces + tmpAdjCnt;
    int nExpextedLamdaSize = tmpAdjCnt * nViews * 2;
    assert(MP.CheckGraph(nExpextedRegions, nExpextedLamdaSize, globalDebugFlag, prompt));

    MP.Solve(nItersMP, flagSaveFullBeliefs, fileAlBeliefs);

    for (int i = 0; i < MP.varState.size(); i++) {
        textureForTriangles[i] = MP.varState[i];
    }

    // delete all pointer to pairwise potentials
    for (auto it: allPairPotentials) delete it;

    MP.DestroyGraph();

    if (globalDebugFlag == 1) {
        std::vector<int> debugTextureForTriangles(nFaces, 0);
        AlignTextureToMeshArgMax(scoreMatrix, debugTextureForTriangles);

        int nDiff = 0;
        for (int f = 0; f < nFaces; f++) {
            if (debugTextureForTriangles[f] != textureForTriangles[f]) nDiff++;
        }

        std::cout << prompt + "[Debug] #diff between argmax and MP: " << nDiff << std::endl;
    }
}

void TextureAligner::CreatePairwisePotential(const int& modelIdx, const int& f1, const int& f2, const int& nViews,
                                             const float& potentialPairVal, const float& potentialPairOffDiagScaleDepth,
                                             const float& potentialPairOffDiagScalePercept, std::vector<float>& potentialPairVec,
                                             const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& validMatrix,
                                             const NS_MeshData::OurMeshData::OurMeshDataStructure& data)
{
    int nActualViews = nViews - 1;

    // get vertice's ID of shared edge
    int p1 = -1, p2 = -1;
    NS_Utils::GetTwoCommonElementsInTriple(data.pointTriangles[modelIdx][f1], data.pointTriangles[modelIdx][f2], p1, p2);
    assert (p1 != -1);
    assert (p2 != -1);

    Eigen::ArrayXXf diffDepth = Eigen::ArrayXXf::Zero(nActualViews, nActualViews);
    Eigen::ArrayXXf diffPerception  = Eigen::ArrayXXf::Zero(nActualViews, nActualViews);

// #pragma omp parallel for schedule(dynamic, 10)
    for (int i = 0; i < nActualViews; i++) {
        // only consider upper triangular
        for (int j = i + 1; j < nActualViews; j++) {
            // We care about situation where faces are valid in different views and only place penalties on them.
            // For situations where (face, camera) are not valid, we just place 0 in pairwise potential
            // since unary will take care for them via giving large penalty. (check function ComputeScoreMatrix)
            if ((validMatrix(i, f1) && validMatrix(j, f2)) || (validMatrix(i, f2) && validMatrix(j, f1))) {

                // get depth difference in all views
                diffDepth(i, j) = std::abs(texAlignerData.vertexDepths[p1][i] - texAlignerData.vertexDepths[p1][j]) + std::abs(texAlignerData.vertexDepths[p2][i] - texAlignerData.vertexDepths[p2][j]);
                diffDepth(j, i) = diffDepth(i, j);

                // get perception difference in all views
                diffPerception(i, j) = (Eigen::Map<const Eigen::Array<float, 1, 3>>(&texAlignerData.vertexColors[p1][i][0]) - Eigen::Map<const Eigen::Array<float, 1, 3>>(&texAlignerData.vertexColors[p1][j][0])).abs().sum() + \
                                       (Eigen::Map<const Eigen::Array<float, 1, 3>>(&texAlignerData.vertexColors[p2][i][0]) - Eigen::Map<const Eigen::Array<float, 1, 3>>(&texAlignerData.vertexColors[p2][j][0])).abs().sum();
                diffPerception(j, i) = diffPerception(i, j);
            }
        }
    }

#pragma omp parallel for
    for (int i = 0; i < nActualViews; i++) {
        for (int j = 0; j < nActualViews; j++) {
            if (i == j)
            {
                // Note, we need to count with nViews, which includes a dummy one, to get the correct index
                potentialPairVec[i * nViews + i] = potentialPairVal;
            }
            else
            {
                potentialPairVec[i * nViews + j] = potentialPairOffDiagScaleDepth * diffDepth(i, j) + potentialPairOffDiagScalePercept * diffPerception(i, j);
            }
        }
    }
}


void TextureAligner::GenRankedFaceCamPairs(const std::vector<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>>& vecValidMatrix,
                                             const std::vector<Eigen::MatrixXf>& vecScoreMatrix,
                                             const std::string& fileFaceCamPairs, const bool& newFile,
                                             const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
                                             const int& globalDebugFlag, const std::string& prompt)
{
    auto start = std::chrono::high_resolution_clock::now();

    int nViews = vecValidMatrix[0].rows();
    // std::vector<std::vector<unsigned short>> texAlignerData.rankedFaceCamPairs(data.nPointTriangles, std::vector<unsigned short>());
    texAlignerData.rankedFaceCamPairs.resize(data.nPointTriangles, std::vector<unsigned short>());

    assert (std::numeric_limits<unsigned short>::max() > nViews);

    for (int k = 0; k < vecValidMatrix.size(); k++) {
        // valid: [#cameras, #faces]; score: [#cameras + 1, #faces] (+1: for dummy)
        assert (vecValidMatrix[k].rows() == nViews);
        assert (vecScoreMatrix[k].rows() == vecValidMatrix[k].rows() + 1);
        assert (vecScoreMatrix[k].cols() == vecValidMatrix[k].cols());

#pragma omp parallel for
        for (int f = 0; f < vecValidMatrix[k].cols(); f++) {
            std::vector<float> tmpFaceCamScores(nViews, 0.0f);
            Eigen::VectorXf::Map(&tmpFaceCamScores[0], nViews) = -1 * vecScoreMatrix[k](Eigen::seq(0, nViews - 1, 1), f);  // the sort function return ascending order. However, we need descending order.
            std::vector<size_t> sortedViewIdxs = NS_Utils::getSortedIdxes(tmpFaceCamScores);

            // if (texAlignerData.textureForTriangles[data.triangleGlobalIdx[k][f]] != nViews) assert(texAlignerData.textureForTriangles[data.triangleGlobalIdx[k][f]] == sortedViewIdxs[0]);
            if (texAlignerData.textureForTriangles[data.triangleGlobalIdx[k][f]] != nViews) assert(tmpFaceCamScores[texAlignerData.textureForTriangles[data.triangleGlobalIdx[k][f]]] == tmpFaceCamScores[sortedViewIdxs[0]]);

            for (int v = 0; v < nViews; v++) {
                if (vecValidMatrix[k](sortedViewIdxs[v], f)) {
                    texAlignerData.rankedFaceCamPairs[data.triangleGlobalIdx[k][f]].push_back((unsigned short)sortedViewIdxs[v]);
                }
            }

        }
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << prompt + "complete writing ranked face-cam pairs in " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms" << std::endl;
    
    if (globalDebugFlag == 1) {
        unsigned int idxSum = 0;
        for (const auto& vec: texAlignerData.rankedFaceCamPairs) {
            for (const unsigned short& it: vec) idxSum += (unsigned int)it;
        }
        std::cout << prompt + "[Debug] writing ranked face-cam pairs, valid index summation: " << idxSum << std::endl;
    }
}


void TextureAligner::RegulateTextureCoords(const int& k, const int& f, const float& onePixelShiftHeight, const float& onePixelShiftWidth,
                                           const std::vector<int>& dataUVCnts, std::vector<NS_TriangleR::CoordFloat*>& tmpVecMtlCoord,
                                           const NS_MeshData::OurMeshData::OurMeshDataStructure& data)
{
    // For each triangle, there is a minimum rectangle which can cover the triangle.
    // We make the rectangle's top/left inclusive and bottom/right exclusive.
    // NOTE: since we rotation/flip when writing mtl into disk,
    // we need to change the end inclusion when reading binary file later.

    // for vertical
    float maxX = std::max({tmpVecMtlCoord[0]->x, tmpVecMtlCoord[1]->x, tmpVecMtlCoord[2]->x});
    float minX = std::min({tmpVecMtlCoord[0]->x, tmpVecMtlCoord[1]->x, tmpVecMtlCoord[2]->x});
    // for horizontal
    float maxY = std::max({tmpVecMtlCoord[0]->y, tmpVecMtlCoord[1]->y, tmpVecMtlCoord[2]->y});
    float minY = std::min({tmpVecMtlCoord[0]->y, tmpVecMtlCoord[1]->y, tmpVecMtlCoord[2]->y});
           
    Eigen::Vector2f dir1, dir2;
    float area1, area2, area3;

    // try to make triangle cover the largest possible areas
    for (int i = 0; i < tmpVecMtlCoord.size(); i++) {
        // for vertical
        if (tmpVecMtlCoord[i]->x == maxX) {
            tmpVecMtlCoord[i]->x = std::ceil(tmpVecMtlCoord[i]->x - 1.0f);
            texAlignerData.uvPtShifts[k][data.textureTriangles[k][f].c[i] - dataUVCnts[k]](1) = -1 * onePixelShiftHeight;
        }
        else if (tmpVecMtlCoord[i]->x == minX) {tmpVecMtlCoord[i]->x = std::floor(tmpVecMtlCoord[i]->x);}
        else {
            dir1(0) = tmpVecMtlCoord[i]->x - tmpVecMtlCoord[(i + 1) % tmpVecMtlCoord.size()]->x;
            dir1(1) = tmpVecMtlCoord[i]->y - tmpVecMtlCoord[(i + 1) % tmpVecMtlCoord.size()]->y;
            dir2(0) = tmpVecMtlCoord[i]->x - tmpVecMtlCoord[(i + 2) % tmpVecMtlCoord.size()]->x;
            dir2(1) = tmpVecMtlCoord[i]->y - tmpVecMtlCoord[(i + 2) % tmpVecMtlCoord.size()]->y;
            area1 = std::fabs(dir1(0) * dir2(1) - dir1(1) * dir2(0)) / 2.0f;
                
            dir1(0) = std::floor(tmpVecMtlCoord[i]->x) - tmpVecMtlCoord[(i + 1) % tmpVecMtlCoord.size()]->x;
            dir1(1) = tmpVecMtlCoord[i]->y - tmpVecMtlCoord[(i + 1) % tmpVecMtlCoord.size()]->y;
            dir2(0) = std::floor(tmpVecMtlCoord[i]->x) - tmpVecMtlCoord[(i + 2) % tmpVecMtlCoord.size()]->x;
            dir2(1) = tmpVecMtlCoord[i]->y - tmpVecMtlCoord[(i + 2) % tmpVecMtlCoord.size()]->y;
            area2 = std::fabs(dir1(0) * dir2(1) - dir1(1) * dir2(0)) / 2.0f;
             
            dir1(0) = std::ceil(tmpVecMtlCoord[i]->x) - tmpVecMtlCoord[(i + 1) % tmpVecMtlCoord.size()]->x;
            dir1(1) = tmpVecMtlCoord[i]->y - tmpVecMtlCoord[(i + 1) % tmpVecMtlCoord.size()]->y;
            dir2(0) = std::ceil(tmpVecMtlCoord[i]->x) - tmpVecMtlCoord[(i + 2) % tmpVecMtlCoord.size()]->x;
            dir2(1) = tmpVecMtlCoord[i]->y - tmpVecMtlCoord[(i + 2) % tmpVecMtlCoord.size()]->y;
            area3 = std::fabs(dir1(0) * dir2(1) - dir1(1) * dir2(0)) / 2.0f;
               
            if (area2 == std::max({area1, area2, area3})) tmpVecMtlCoord[i]->x = std::floor(tmpVecMtlCoord[i]->x);
            if (area3 == std::max({area1, area2, area3})) tmpVecMtlCoord[i]->x = std::ceil(tmpVecMtlCoord[i]->x);
        }
    
        // for horizontal
        if (tmpVecMtlCoord[i]->y == maxY) {
                tmpVecMtlCoord[i]->y = std::ceil(tmpVecMtlCoord[i]->y - 1.0f);
                texAlignerData.uvPtShifts[k][data.textureTriangles[k][f].c[i] - dataUVCnts[k]](0) = -1 * onePixelShiftWidth;
        }
        else if (tmpVecMtlCoord[i]->y == minY) {tmpVecMtlCoord[i]->y = std::floor(tmpVecMtlCoord[i]->y);}
        else {
            dir1(0) = tmpVecMtlCoord[i]->x - tmpVecMtlCoord[(i + 1) % tmpVecMtlCoord.size()]->x;
            dir1(1) = tmpVecMtlCoord[i]->y - tmpVecMtlCoord[(i + 1) % tmpVecMtlCoord.size()]->y;
            dir2(0) = tmpVecMtlCoord[i]->x - tmpVecMtlCoord[(i + 2) % tmpVecMtlCoord.size()]->x;
            dir2(1) = tmpVecMtlCoord[i]->y - tmpVecMtlCoord[(i + 2) % tmpVecMtlCoord.size()]->y;
            area1 = std::fabs(dir1(0) * dir2(1) - dir1(1) * dir2(0)) / 2.0f;
                
            dir1(0) = tmpVecMtlCoord[i]->x - tmpVecMtlCoord[(i + 1) % tmpVecMtlCoord.size()]->x;
            dir1(1) = std::floor(tmpVecMtlCoord[i]->y) - tmpVecMtlCoord[(i + 1) % tmpVecMtlCoord.size()]->y;
            dir2(0) = tmpVecMtlCoord[i]->x - tmpVecMtlCoord[(i + 2) % tmpVecMtlCoord.size()]->x;
            dir2(1) = std::floor(tmpVecMtlCoord[i]->y) - tmpVecMtlCoord[(i + 2) % tmpVecMtlCoord.size()]->y;
            area2 = std::fabs(dir1(0) * dir2(1) - dir1(1) * dir2(0)) / 2.0f;
                
            dir1(0) = tmpVecMtlCoord[i]->x - tmpVecMtlCoord[(i + 1) % tmpVecMtlCoord.size()]->x;
            dir1(1) = std::ceil(tmpVecMtlCoord[i]->y) - tmpVecMtlCoord[(i + 1) % tmpVecMtlCoord.size()]->y;
            dir2(0) = tmpVecMtlCoord[i]->x - tmpVecMtlCoord[(i + 2) % tmpVecMtlCoord.size()]->x;
            dir2(1) = std::ceil(tmpVecMtlCoord[i]->y) - tmpVecMtlCoord[(i + 2) % tmpVecMtlCoord.size()]->y;
            area3 = std::fabs(dir1(0) * dir2(1) - dir1(1) * dir2(0)) / 2.0f;
              
            if (area2 == std::max({area1, area2, area3})) tmpVecMtlCoord[i]->y = std::floor(tmpVecMtlCoord[i]->y);
            if (area3 == std::max({area1, area2, area3})) tmpVecMtlCoord[i]->y = std::ceil(tmpVecMtlCoord[i]->y);
        }
    }
}


void TextureAligner::GenCompactMaterialImage(const bool& flagRePack, const bool& flagInterpolate, const bool& flagAverage,
                                             const int& planeIdx, const int& mtlWidth, const int& mtlHeight,
                                             NS_StreamFile::StreamFile& streamInfos,
                                             const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
                                             NS_ImgWriter::ImgWriter& imgWriter, const NS_ImgWriter::ImgProcessTypeEnum& writerImgProcessType,
                                             const bool& flagConformalMap, const int& nExtrudePixels,
                                             const std::string& mtlImgFile, const std::string& mtlCoverImgFile,
                                             const int& globalDebugFlag, const std::string& prompt)
{
    auto tex_start = std::chrono::high_resolution_clock::now();

    boost::gil::rgb8_image_t mtlRGBImg(mtlWidth, mtlHeight);
    auto mtlRGBImgView = boost::gil::view(mtlRGBImg);

    // placeholder
    Eigen::MatrixXf mtlTriLocalCoord1 = Eigen::MatrixXf::Zero(1, 1);
    Eigen::MatrixXf mtlTriLocalCoord2 = Eigen::MatrixXf::Zero(1, 1);

    std::string localPrompt = prompt + "[plane " + std::to_string(planeIdx) + "/" + std::to_string(data.maxPlaneIdx) + "] ";

    const std::vector<std::vector<Eigen::Vector2d>>& dataUVPts = flagRePack ? data.uvPtsAfterRePack : data.uvPts;
    const std::vector<int>& dataUVCnts = flagRePack ? data.uvCntsAfterRePack : data.uvCnts;

    // initialize uv coordinates shift to avoid overlapping
    texAlignerData.uvPtShifts.resize(dataUVPts.size());
    for (int k = 0; k < dataUVPts.size(); k++) {
        texAlignerData.uvPtShifts[k].resize(dataUVPts[k].size(), Eigen::Vector2d::Zero());
    }
    float onePixelShiftWidth = 1.0f / mtlWidth;
    float onePixelShiftHeight = 1.0f / mtlHeight;

    std::vector<char> globalFlags(mtlWidth * mtlHeight, 0);

#pragma omp parallel 
{
    std::vector<char> localFlags(mtlWidth * mtlHeight, 0);

#pragma omp for
    for (int k = 0; k < data.textureTriangles.size(); k++) { //loop over mesh components
        for(int f = 0; f < data.textureTriangles[k].size(); ++f) { //loop over faces in components

            bool flag;
            if (flagRePack) flag = (data.planeIdxAfterRePack[data.triangleGlobalIdx[k][f]] == planeIdx);
            else flag = (data.planeIdxAfterMP[data.triangleGlobalIdx[k][f]] == planeIdx);

            if (flag) {

                // cur_step += 1;
    
                auto tex_inner_start = std::chrono::high_resolution_clock::now();
    
                int textureIX = texAlignerData.textureForTriangles[data.triangleGlobalIdx[k][f]];
    
                // need to check whether dummy texture is assigned
                if (textureIX < streamInfos.NumberOfEntries()) {
    
                    const boost::gil::rgb8_image_t* const rgbImg = streamInfos.GetRGBImage(textureIX);
    
                    // NOTE: need to check which one x-axis stands for
                    Eigen::Matrix<float, 2, 3> imgCoords = texAlignerData.NDC(Eigen::seq(2 * textureIX, 2 * textureIX + 1, 1), data.pointTriangles[k][f].c);
                    // NOTE: Important !!! Must use realheight here
                    imgCoords.row(0) = imgCoords.row(0).array() * rgbImg->height();
                    imgCoords.row(1) = imgCoords.row(1).array() * rgbImg->width();
    
                    // NOTE: in NS_TriangleR::CoordFloat struct, x is for vertical (height), aligned with 1st elem of texAlignerData.NDC 
                    NS_TriangleR::CoordFloat streamImgCoord1 = {imgCoords(0, 0), imgCoords(1, 0)};
                    NS_TriangleR::CoordFloat streamImgCoord2 = {imgCoords(0, 1), imgCoords(1, 1)};
                    NS_TriangleR::CoordFloat streamImgCoord3 = {imgCoords(0, 2), imgCoords(1, 2)};
    
                    // remember to use local indices
                    // NS_TriangleR::CoordFloat.x is for vertical (height), u (1st elem of uvPts) is for horizontal
                    NS_TriangleR::CoordFloat mtlCoord1, mtlCoord2, mtlCoord3;
                   
                    mtlCoord1 = {(float)dataUVPts[k][data.textureTriangles[k][f].c[0] - dataUVCnts[k]](1) * mtlHeight,
                                 (float)dataUVPts[k][data.textureTriangles[k][f].c[0] - dataUVCnts[k]](0) * mtlWidth};
                    mtlCoord2 = {(float)dataUVPts[k][data.textureTriangles[k][f].c[1] - dataUVCnts[k]](1) * mtlHeight,
                                 (float)dataUVPts[k][data.textureTriangles[k][f].c[1] - dataUVCnts[k]](0) * mtlWidth};
                    mtlCoord3 = {(float)dataUVPts[k][data.textureTriangles[k][f].c[2] - dataUVCnts[k]](1) * mtlHeight,
                                 (float)dataUVPts[k][data.textureTriangles[k][f].c[2] - dataUVCnts[k]](0) * mtlWidth};
                    
                    if (!flagConformalMap) {
                        std::vector<NS_TriangleR::CoordFloat*> tmpVecMtlCoord{&mtlCoord1, &mtlCoord2, &mtlCoord3};
                        RegulateTextureCoords(k, f, onePixelShiftHeight, onePixelShiftWidth, dataUVCnts, tmpVecMtlCoord, data);
                    }
    
                    if (flagInterpolate) 
                    {
                        NS_TriangleRI::TriangleRasterizationInterpolate::FillTriangle(
                            mtlCoord1, mtlCoord2, mtlCoord3, streamImgCoord1, streamImgCoord2, streamImgCoord3,
                            localFlags, textureIX, boost::gil::const_view(*rgbImg), mtlRGBImgView);
                    } 
                    else if (flagAverage) 
                    {
                        NS_TriangleRA::TriangleRasterizationAverage::FillTriangle(
                            mtlCoord1, mtlCoord2, mtlCoord3, streamImgCoord1, streamImgCoord2, streamImgCoord3,
                            localFlags, textureIX, boost::gil::const_view(*rgbImg), mtlRGBImgView);
                        
                    } else {
                        NS_TriangleR::TriangleRasterization::FillTriangle(
                            mtlCoord1, mtlCoord2, mtlCoord3, streamImgCoord1, streamImgCoord2, streamImgCoord3,
                            localFlags, globalFlags, textureIX, boost::gil::const_view(*rgbImg), mtlRGBImgView, mtlTriLocalCoord1, mtlTriLocalCoord2,
                            true, nExtrudePixels);
                    }
                    auto tex_inner_finish = std::chrono::high_resolution_clock::now();

                }
            }
        }
    }
}

    imgWriter.AddImageToQueue(mtlRGBImg, mtlImgFile, writerImgProcessType);

    // write coverage mtl image
    boost::gil::rgb8_image_t mtlCoverageImg(mtlWidth, mtlHeight);
    auto mtlCoverageView = boost::gil::view(mtlCoverageImg);
    for (int i = 0; i < mtlCoverageView.height(); i++) {
        for (int j = 0; j < mtlCoverageView.width(); j++) {
            if (globalFlags[i * mtlWidth + j] > 0) {
                // NOTE: boost::gil's 1st coordiante is for horizontal
                mtlCoverageView(j, i) = boost::gil::rgb8_pixel_t{255, 255, 255};
            }
        }
    }
    // boost::gil::write_view(mtlCoverImgFile, boost::gil::const_view(mtlCoverageImg), boost::gil::png_tag());
    imgWriter.AddImageToQueue(mtlCoverageImg, mtlCoverImgFile, writerImgProcessType);
    
    auto tex_finish = std::chrono::high_resolution_clock::now();
    std::cout << localPrompt + "... complete generating material files: " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(tex_finish - tex_start).count() << " ms" << std::endl;
    
    if (globalDebugFlag == 1) {
        int debugTexMtlSum = 0;
        for (auto it: globalFlags) debugTexMtlSum += it;
        std::cout << prompt + "[Debug] texture on mtl sum: " << debugTexMtlSum << std::endl;
    }
}


void TextureAligner::GenCompactMaterialBinaryFile(const bool& flagRePack, const int& planeIdx,
                                                  const int& mtlWidth, const int& mtlHeight,
                                                  NS_StreamFile::StreamFile& streamInfos,
                                                  const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
                                                  NS_ImgWriter::ImgWriter& imgWriter, const NS_ImgWriter::ImgProcessTypeEnum& writerImgProcessType,
                                                  const bool& flagTopOneAssignedMtl, const int& nExtrudePixels,
                                                  const std::string& fileMtlBin, const bool& newFile, const std::string& rawDir,
                                                  const int& globalDebugFlag, const std::string& prompt)
{
    auto tex_start = std::chrono::high_resolution_clock::now();

    // placeholder
    boost::gil::rgb8_image_t mtlRGBImg(1, 1);
    auto mtlRGBImgView = boost::gil::view(mtlRGBImg);
    int textureIX = 0;

    Eigen::MatrixXf mtlTriLocalCoord1 = Eigen::MatrixXf::Constant(mtlWidth, mtlHeight, -1.0f);
    Eigen::MatrixXf mtlTriLocalCoord2 = Eigen::MatrixXf::Constant(mtlWidth, mtlHeight, -1.0f);

    const std::vector<std::vector<Eigen::Vector2d>>& dataUVPts = flagRePack ? data.uvPtsAfterRePack : data.uvPts;
    const std::vector<int>& dataUVCnts = flagRePack ? data.uvCntsAfterRePack : data.uvCnts;

    // initialize uv coordinates shift to avoid overlapping
    texAlignerData.uvPtShifts.resize(dataUVPts.size());
    for (int k = 0; k < dataUVPts.size(); k++) {
        texAlignerData.uvPtShifts[k].resize(dataUVPts[k].size(), Eigen::Vector2d::Zero());
    }
    float onePixelShiftWidth = 1.0f / mtlWidth;
    float onePixelShiftHeight = 1.0f / mtlHeight;

    std::string localPrompt = prompt + "[plane " + std::to_string(planeIdx) + "/" + std::to_string(data.maxPlaneIdx) + "] ";

    std::vector<char> globalFlags(mtlWidth * mtlHeight, 0);

#pragma omp parallel 
{
    std::vector<char> localFlags(mtlWidth * mtlHeight, 0);
#pragma omp for
    for (int k = 0; k < data.textureTriangles.size(); k++) { //loop over mesh components
        for(int f = 0; f < data.textureTriangles[k].size(); ++f) { //loop over faces in components

            bool flag;
            if (flagRePack) flag = (data.planeIdxAfterRePack[data.triangleGlobalIdx[k][f]] == planeIdx);
            else flag = (data.planeIdxAfterMP[data.triangleGlobalIdx[k][f]] == planeIdx);

            if (flag) {

                // This is a placeholder
                NS_TriangleR::CoordFloat streamImgCoord1 = {0, 0};
                NS_TriangleR::CoordFloat streamImgCoord2 = {0, 0};
                NS_TriangleR::CoordFloat streamImgCoord3 = {0, 0};

                // remember to use local indices
                // NS_TriangleR::CoordFloat.x is for vertical (height), u (1st elem of uvPts) is for horizontal
                NS_TriangleR::CoordFloat mtlCoord1, mtlCoord2, mtlCoord3;

                mtlCoord1 = {(float)dataUVPts[k][data.textureTriangles[k][f].c[0] - dataUVCnts[k]](1) * mtlHeight,
                             (float)dataUVPts[k][data.textureTriangles[k][f].c[0] - dataUVCnts[k]](0) * mtlWidth};
                mtlCoord2 = {(float)dataUVPts[k][data.textureTriangles[k][f].c[1] - dataUVCnts[k]](1) * mtlHeight,
                             (float)dataUVPts[k][data.textureTriangles[k][f].c[1] - dataUVCnts[k]](0) * mtlWidth};
                mtlCoord3 = {(float)dataUVPts[k][data.textureTriangles[k][f].c[2] - dataUVCnts[k]](1) * mtlHeight,
                             (float)dataUVPts[k][data.textureTriangles[k][f].c[2] - dataUVCnts[k]](0) * mtlWidth};

                std::vector<NS_TriangleR::CoordFloat*> tmpVecMtlCoord{&mtlCoord1, &mtlCoord2, &mtlCoord3};

                RegulateTextureCoords(k, f, onePixelShiftHeight, onePixelShiftWidth, dataUVCnts, tmpVecMtlCoord, data);
                    
                NS_TriangleR::TriangleRasterization::FillTriangle(
                    mtlCoord1, mtlCoord2, mtlCoord3, streamImgCoord1, streamImgCoord2, streamImgCoord3,
                    localFlags, globalFlags, textureIX, boost::gil::const_view(mtlRGBImg), mtlRGBImgView,
                    mtlTriLocalCoord1, mtlTriLocalCoord2, flagTopOneAssignedMtl, nExtrudePixels);
            }
        }
    }
}

    if (writerImgProcessType == NS_ImgWriter::ImgProcessTypeEnum::TRANSPOSE_ROT90CCW) {

        // flip up/down
        mtlTriLocalCoord1.colwise().reverseInPlace();
        // mtlTriLocalCoord1.transposeInPlace();
        mtlTriLocalCoord2.colwise().reverseInPlace();
        // mtlTriLocalCoord2.transposeInPlace();
    }

    unsigned int rows = mtlTriLocalCoord1.rows(), cols = mtlTriLocalCoord1.cols();
    assert (std::numeric_limits<unsigned int>::max() > rows);
    assert (std::numeric_limits<unsigned int>::max() > cols);
    
    std::ofstream outFileStream;
    outFileStream.open(fileMtlBin, std::ios_base::binary | std::ios_base::out);
    
    if (!outFileStream.is_open()) {
        std::cout << "Unable to open binary file for saving material information." << std::endl;
        exit(EXIT_FAILURE);
    }

    outFileStream.write((char*)(&rows), sizeof(unsigned int));
    outFileStream.write((char*)(&cols), sizeof(unsigned int));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            outFileStream.write((char*)&mtlTriLocalCoord1(i, j), sizeof(float));
        }
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            outFileStream.write((char*)&mtlTriLocalCoord2(i, j), sizeof(float));
        }
    }
    outFileStream.close();
    
    auto tex_finish = std::chrono::high_resolution_clock::now();
    std::cout << localPrompt + "... complete generating material binary files: " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(tex_finish - tex_start).count() << " ms" << std::endl;
            
    if (globalDebugFlag == 1) {
        std::cout << localPrompt + "[Debug] mtl tri local coordinate 1st sum: " << mtlTriLocalCoord1.array().sum() \
                  << "; 2nd sum: " << mtlTriLocalCoord2.array().sum() << std::endl;
    }

}


void TextureAligner::GenNonCompactMaterialImage(NS_StreamFile::StreamFile& streamInfos,
                                                 const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
                                                 NS_ImgWriter::ImgWriter& imgWriter, const NS_ImgWriter::ImgProcessTypeEnum& writerImgProcessType,
                                                 std::vector<bool>& flagProcessedStreamImg, const std::string& outDir,
                                                 const int& globalDebugFlag, const std::string& prompt)
{
    if (texAlignerData.flagNeedStreamImg.size() == 0) {
        // this function is called for the first time, initialize the flags
        texAlignerData.flagNeedStreamImg.resize(streamInfos.NumberOfEntries(), false);
    }

    std::string localPrompt = prompt;

    // std::cout << localPrompt + "Start generating material files with " << total_steps << " triangles ..." << std::endl;

    auto texStart = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < data.pointTriangles.size(); k++) { // loop over mesh components
        for (int f = 0; f < data.pointTriangles[k].size(); ++f) { // loop over faces in components
            
            int textureIX = texAlignerData.textureForTriangles[data.triangleGlobalIdx[k][f]];

            // this image has been written to disk already
            if (flagProcessedStreamImg[textureIX]) texAlignerData.flagNeedStreamImg[textureIX] = true;

            // need to check whether dummy texture is assigned
            if ((textureIX < streamInfos.NumberOfEntries()) && (!texAlignerData.flagNeedStreamImg[textureIX]) ) {
                texAlignerData.flagNeedStreamImg[textureIX] = true;
                const boost::gil::rgb8_image_t* const rgbImg = streamInfos.GetRGBImage(textureIX);
                std::string mtlImgFile = outDir + "/mtl_" + std::to_string(textureIX) + ".png";
                imgWriter.AddImageToQueue(*rgbImg, mtlImgFile, writerImgProcessType);
            }
        }
    }

    auto texFinish = std::chrono::high_resolution_clock::now();
    std::cout << localPrompt + "... complete generating material files: " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(texFinish - texStart).count() << " ms" << std::endl;
}


void TextureAligner::ObjFileWriter(const bool& flagRePack, std::ofstream& outf, std::ofstream& fcPairStream,
                                   const int& cumFaceCnt, const int& cumVertexCnt, const int& cumTexCnt, nlohmann::json& jsonFaceID,
                                   const std::string& mtlIndex, const bool& flagTopOneAssignedMtl, const bool& flagNoMtlVt,
                                   const NS_MeshData::OurMeshData::OurMeshDataStructure& data)
{
    // write all 3D points
    for(int i = 0; i < data.Points.cols(); ++i) {
        outf << "v " << data.Points(0, i) << " " << data.Points(1, i) << " " << data.Points(2, i) << std::endl;
    }

    if (!flagNoMtlVt) {
        // write all 2D texture points
        if (flagRePack)
        {
            for(int k = 0; k < data.uvPtsAfterRePack.size(); ++k) {
                for(int i = 0; i < data.uvPtsAfterRePack[k].size(); ++i) {
                    outf << "vt " << std::max({0.0, data.uvPtsAfterRePack[k][i](0) + texAlignerData.uvPtShifts[k][i](0)}) << " " \
                         << std::max({0.0, data.uvPtsAfterRePack[k][i](1) + texAlignerData.uvPtShifts[k][i](1)}) << std::endl;
                }
            }
        }
        else
        {
            for(int k = 0; k < data.uvPts.size(); ++k) {
                for(int i = 0; i < data.uvPts[k].size(); ++i) {
                    outf << "vt " << data.uvPts[k][i](0) << " " << data.uvPts[k][i](1) << std::endl;
                }
            }
        }
    }

    int faceCnt = 0;
    for (int planeIdx = 0; planeIdx <= data.maxPlaneIdx; planeIdx++) {
        outf << "usemtl mtl" << mtlIndex + "_" + std::to_string(planeIdx) << std::endl;
        // NOTE: indices are 1-based and index is global consistent
        for(int k = 0; k < data.pointTriangles.size(); ++k) {
            for (int f = 0; f < data.pointTriangles[k].size(); f++) {
                bool flag;
                if (flagRePack) flag = (data.planeIdxAfterRePack[data.triangleGlobalIdx[k][f]] == planeIdx);
                else flag = (data.planeIdxAfterMP[data.triangleGlobalIdx[k][f]] == planeIdx);

                if (flag) {

                    jsonFaceID[std::to_string(cumFaceCnt + data.triangleGlobalIdx[k][f])] = cumFaceCnt + faceCnt;
                    faceCnt++;

                    if (flagNoMtlVt)
                    {
                        outf << "f" \
                             << " " << data.pointTriangles[k][f].c[0] + 1 + cumVertexCnt \
                             << " " << data.pointTriangles[k][f].c[1] + 1 + cumVertexCnt \
                             << " " << data.pointTriangles[k][f].c[2] + 1 + cumVertexCnt << std::endl;   
                    }
                    else
                    {
                        outf << "f" \
                            << " " << data.pointTriangles[k][f].c[0] + 1 + cumVertexCnt \
                            << "/" << data.textureTriangles[k][f].c[0] + 1 + cumTexCnt  \
                            << " " << data.pointTriangles[k][f].c[1] + 1 + cumVertexCnt \
                            << "/" << data.textureTriangles[k][f].c[1] + 1 + cumTexCnt  \
                            << " " << data.pointTriangles[k][f].c[2] + 1 + cumVertexCnt \
                            << "/" << data.textureTriangles[k][f].c[2] + 1 + cumTexCnt << std::endl;   
                        
                        if (!flagTopOneAssignedMtl) {
                            unsigned short vecSize = (unsigned short)texAlignerData.rankedFaceCamPairs[data.triangleGlobalIdx[k][f]].size();
                            fcPairStream.write((const char*)&vecSize, sizeof(unsigned short));
                            if (vecSize != 0) {
                                for (const unsigned short& it: texAlignerData.rankedFaceCamPairs[data.triangleGlobalIdx[k][f]]) {
                                    fcPairStream.write((const char*)&it, sizeof(unsigned short));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    assert (faceCnt == data.nPointTriangles);
}


void TextureAligner::WriteObjMtlFilesCompactMtl(const bool& flagRePack, const std::string& objFile, const std::string& mtlFile,
                                                const bool& newFile, const std::string& mtlIndex,
                                                const int& cumFaceCnt, const int& cumVertexCnt, const int& cumTexCnt,
                                                const bool& flagTopOneAssignedMtl, const bool& flagNoMtlVt,
                                                const std::string& fileFaceCamPairs, const std::string& fileMeshFaceID2ObjFaceID,
                                                const NS_MeshData::OurMeshData::OurMeshDataStructure& data)
{
    std::ofstream mtlFileStream, objFileStream, fcPairStream;

    // write material file
    if (newFile){
        mtlFileStream.open(mtlFile, std::ios_base::out);
    } else {
        mtlFileStream.open(mtlFile, std::ios_base::app);
    }
    if (!mtlFileStream.is_open()) {
        std::cout << "Unable to open material file." << std::endl;
        exit(EXIT_FAILURE);
    }
    for (int planeIdx = 0; planeIdx <= data.maxPlaneIdx; planeIdx ++) {
        mtlFileStream << "newmtl mtl" << mtlIndex + "_" + std::to_string(planeIdx) << std::endl \
                      << "    Ka 1.000 1.000 1.000" << std::endl \
                      << "    Kd 1.000 1.000 1.000" << std::endl \
                      << "    Ks 0.000 0.000 0.000" << std::endl \
                      << "    d 1.0" << std::endl \
                      << "    illum 2" << std::endl \
                      << "    map_Ka mtl_" << mtlIndex + "_" + std::to_string(planeIdx) << ".png" << std::endl \
                      << "    map_Kd mtl_" << mtlIndex + "_" + std::to_string(planeIdx) << ".png" << std::endl;
    }
    mtlFileStream.close();

    // write obj file
    if (newFile){
        objFileStream.open(objFile, std::ios_base::out);
        objFileStream << "mtllib TexAlign.mtl" << std::endl;
    } else {
        objFileStream.open(objFile, std::ios_base::app);
    }
    if (!objFileStream.is_open()) {
        std::cout << "Unable to open .obj file." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (!flagTopOneAssignedMtl) {
        if (newFile) fcPairStream.open(fileFaceCamPairs, std::ios_base::binary | std::ios_base::out);
        else fcPairStream.open(fileFaceCamPairs, std::ios_base::binary | std::ios_base::app);
        
        if (!fcPairStream.is_open()) {
            std::cout << "Unable to open binary file for saving ranked face-camera pairs." << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    nlohmann::json jsonFaceID;
    if (!newFile) {
        std::ifstream ifs(fileMeshFaceID2ObjFaceID);
        if (!ifs.is_open()) {
            std::cout << "Unable to open face id conversion json file." << std::endl;
            exit(EXIT_FAILURE);
        }
        jsonFaceID = nlohmann::json::parse(ifs);
        ifs.close();
    }

    objFileStream << "g _" << mtlIndex << std::endl;
    ObjFileWriter(flagRePack, objFileStream, fcPairStream, cumFaceCnt, cumVertexCnt, cumTexCnt, jsonFaceID, mtlIndex, flagTopOneAssignedMtl, flagNoMtlVt, data);
    objFileStream << "s off" << std::endl;
    objFileStream.close();
    fcPairStream.close();

    std::ofstream jsonFaceStream(fileMeshFaceID2ObjFaceID, std::ios_base::out);
    if (!jsonFaceStream.is_open()) {
        std::cout << "Unable to open face id conversion json file." << std::endl;
        exit(EXIT_FAILURE);
    }
    jsonFaceStream << jsonFaceID << std::endl;
    jsonFaceStream.close();
}


void TextureAligner::WriteObjMtlFilesNonCompactMtl(const std::string& groudIdx, const std::string& objFile,
                                                 const std::string& mtlFile, const bool& newFile,
                                                 std::vector<bool>& flagProcessedStreamImg,
                                                 const int& cumVertexCnt, const int& cumTexCnt, int& curTexCnt,
                                                 const NS_MeshData::OurMeshData::OurMeshDataStructure& data)
{
    std::ofstream mtlFileStream, objFileStream;

    if (newFile){
        mtlFileStream.open(mtlFile, std::ios_base::out);

        objFileStream.open(objFile, std::ios_base::out);
        objFileStream << "mtllib TexAlign.mtl" << std::endl;
    } else {
        mtlFileStream.open(mtlFile, std::ios_base::app);
        objFileStream.open(objFile, std::ios_base::app);
    }

    if (!mtlFileStream.is_open()) {
        std::cout << "Unable to open material file." << std::endl;
        exit(EXIT_FAILURE);
    }
    if (!objFileStream.is_open()) {
        std::cout << "Unable to open .obj file." << std::endl;
        exit(EXIT_FAILURE);
    }

    // this nViews does not count the dummy view, which has the largest index
    int nViews = texAlignerData.NDC.rows() / 2;

    objFileStream << "g _" << groudIdx << std::endl;

    // write all 3D points
    for (int i = 0; i < data.Points.cols(); ++i) {
        objFileStream << "v " << data.Points(0, i) << " " << data.Points(1, i) << " " << data.Points(2, i) << std::endl;
    }

    // a map stores the following {3D_point_idx: {viewId1: vtIdx1, viewId2: vtIdx2, ...}}
    // the main target is to avoid duplication of 2D texture points
    std::map<unsigned int, std::map<unsigned int, unsigned int>> vt2MtlIdx;

    // write all texture coordinates
    for(int k = 0; k < data.pointTriangles.size(); ++k) {
        for (int f = 0; f < data.pointTriangles[k].size(); f++) {
            int textureIdx = texAlignerData.textureForTriangles[data.triangleGlobalIdx[k][f]]; 

            if (textureIdx != nViews) {
                // not a dummy texture
                for (int i = 0; i < data.pointTriangles[k][f].length; i++) {
                    unsigned int v = data.pointTriangles[k][f].c[i];
                    if (vt2MtlIdx.find(v) == vt2MtlIdx.end()) vt2MtlIdx[v] = std::map<unsigned int, unsigned int>();  // initialize the value
    
                    if (vt2MtlIdx[v].find(textureIdx) == vt2MtlIdx[v].end()) {
                        vt2MtlIdx[v][textureIdx] = curTexCnt;
                        curTexCnt += 1;
                        objFileStream << "vt " << texAlignerData.NDC(2 * textureIdx, v) << " " << texAlignerData.NDC(2 * textureIdx + 1, v) << std::endl;
                    }
                }
            }
        }
    }

    // write all faces
    int prevTexIdx = -1;
    int curTexIdx;

    for(int k = 0; k < data.pointTriangles.size(); ++k) {
        for (int f = 0; f < data.pointTriangles[k].size(); f++) {

            curTexIdx = texAlignerData.textureForTriangles[data.triangleGlobalIdx[k][f]];
        
            if (curTexIdx != prevTexIdx) {
                prevTexIdx = curTexIdx;
                objFileStream << "usemtl mtl" << curTexIdx << std::endl;

                if ((curTexIdx < nViews) && (!flagProcessedStreamImg[curTexIdx])) {
                    flagProcessedStreamImg[curTexIdx] = true;
                    mtlFileStream << "newmtl mtl" << curTexIdx << std::endl \
                                  << "    Ka 1.000 1.000 1.000" << std::endl \
                                  << "    Kd 1.000 1.000 1.000" << std::endl \
                                  << "    Ks 0.000 0.000 0.000" << std::endl \
                                  << "    d 1.0" << std::endl \
                                  << "    illum 2" << std::endl \
                                  << "    map_Ka mtl_" << curTexIdx << ".png" << std::endl \
                                  << "    map_Kd mtl_" << curTexIdx << ".png" << std::endl;
                }
            }

            // obj file is 1-based indexing
            if (curTexIdx == nViews) {
                // dummy texture
                objFileStream << "f " \
                              << " " << data.pointTriangles[k][f].c[0] + 1 + cumVertexCnt \
                              << " " << data.pointTriangles[k][f].c[1] + 1 + cumVertexCnt \
                              << " " << data.pointTriangles[k][f].c[2] + 1 + cumVertexCnt << std::endl;

            } else {
                objFileStream << "f" \
                              << " " << data.pointTriangles[k][f].c[0] + 1 + cumVertexCnt \
                              << "/" << vt2MtlIdx[data.pointTriangles[k][f].c[0]][curTexIdx] + 1 + cumTexCnt \
                              << " " << data.pointTriangles[k][f].c[1] + 1 + cumVertexCnt \
                              << "/" << vt2MtlIdx[data.pointTriangles[k][f].c[1]][curTexIdx] + 1 + cumTexCnt \
                              << " " << data.pointTriangles[k][f].c[2] + 1 + cumVertexCnt \
                              << "/" << vt2MtlIdx[data.pointTriangles[k][f].c[2]][curTexIdx] + 1 + cumTexCnt << std::endl;
            }
        }
    }

    objFileStream << "s off" << std::endl;

    mtlFileStream.close();
    objFileStream.close();       
}

}  /* end namespace */