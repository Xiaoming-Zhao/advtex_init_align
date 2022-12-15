#ifndef H_TEXTURE_ALIGNER
#define H_TEXTURE_ALIGNER

#include <vector>
#include <string>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <nlohmann/json.hpp>

#include "StreamFile.h"
#include "OurMeshData.h"
#include "MPSolver.h"
#include "ImgWriter.h"
#include "TriangleRasterization.h"
#include "TriangleRasterizationInterpolate.h"
#include "TriangleRasterizationAverage.h"

namespace NS_TextureAlign {


#define MAX_VAL_TO_AVOID_NAN 999999.f
#define DEPTH_DISCREPANCY_ABS_TOL 0.25  // NOTE: this was originally used for depth from Apple since the LIDAR only supports up to 5 meters
#define DEPTH_DISCREPANCY_REL_TOL 0.1
#define DEPTH_DISCREPANCY_MAX_PENALTY 100

class TextureAligner {

public:
    enum TextureAlignmentApproach {
        ArgMax = 0,
        MessagePassing = 1,
    };

    struct TexAlignerDataStructure {
        Eigen::MatrixXf vertexHomoCoords;
        Eigen::MatrixXf NDC;  // [2 * #cameras, #points]
        // Eigen::MatrixXf faceCenterNDC;
        std::vector<int> homoCoordNDCIdx;
    
        std::vector<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>> faceVertexVisibilityMatrix;
        std::vector<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>> faceCenterPostiveZMatrix;
        std::vector<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>> faceCenterVisibilityMatrix;
        std::vector<Eigen::MatrixXf> faceAreaMatrix;
        std::vector<Eigen::MatrixXf> faceCameraDistMatrix;
        std::vector<Eigen::MatrixXf> faceCameraDistRefMatrix;
        std::vector<Eigen::MatrixXf> faceVertexPerceptionConsistencyMatrix;
    
        std::vector<std::map<int, float>> vertexDepths;
    
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> validVertexCameraPair;
        // int vertexColorTupleLen = 0;
        std::vector<std::map<int, std::vector<float>>> vertexColors;  // every vertexColorTupleLen elements represent an RGB
        std::vector<std::vector<float>> meanVertexColors;  // every three elements represent an RGB
        std::vector<Eigen::MatrixXf> faceColorMatrix;
    
        std::vector<int> textureForTriangles;
        std::vector<std::vector<unsigned short>> rankedFaceCamPairs;
    
        std::vector<std::vector<Eigen::Vector2d>> uvPtShifts;
    
        std::vector<bool> flagNeedStreamImg;
    };
public:
    TextureAligner();
    ~TextureAligner();

    void ClearCues();

    const TexAlignerDataStructure& GetData() const;
    const Eigen::MatrixXf& GetNDC() const;
    const std::vector<Eigen::MatrixXf> GetValidFaceAreaMatrix();

    // const std::vector<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>>& GetValidMatrix() const;

    void ComputeNDC(const NS_StreamFile::StreamFile& streamInfos,
                    const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
                    // const bool& flagTopOneAssignedMtl, const std::string& fileNDC, const bool& newFile,
                    const int& globalDebugFlag, const std::string& prompt);
    void ComputeCues(NS_StreamFile::StreamFile& streamInfos,
                     const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
                     const bool& flagFaceAreaNDC,
                     const int& globalDebugFlag, const std::string& prompt);

    void WriteBin(const std::string& binDir, const bool& newFile, const int& globalDebugFlag, const std::string& prompt);

    void AlignTextureToMesh(const int& approachType,
                            const float& unaryPotentialDummy, const float& penaltyFaceArea,
                            const float& penaltyFaceCamDist, const float& penaltyFaceVertexPerceptionConsist,
                            NS_MPSolver::MPSolver<float, int>& MP, const int& nItersMP,
                            const float& pairwisePotentialMP, const float& pairwisePotentialOffDiagScaleDepth, const float& pairwisePotentialOffDiagScalePercept,
                            const bool& flagSaveFullBeliefs, const std::string& fileAlBeliefs,
                            const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
                            const bool& flagTopOneAssignedMtl, const std::string& fileFaceCamPairs, const bool& newFile,
                            const int& globalDebugFlag, const std::string& prompt);

    void GenCompactMaterialImage(const bool& flagRePack, const bool& flagInterpolate, const bool& flagAverage,
                                 const int& planeIdx, const int& mtlWidth, const int& mtlHeight,
                                 NS_StreamFile::StreamFile& streamInfos,
                                 const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
                                 NS_ImgWriter::ImgWriter& imgWriter, const NS_ImgWriter::ImgProcessTypeEnum& writerImgProcessType,
                                 const bool& flagConformalMap, const int& nExtrudePixels,
                                 const std::string& mtlImgFile, const std::string& mtlCoverImgFile,
                                 const int& globalDebugFlag, const std::string& prompt);
    void GenCompactMaterialBinaryFile(const bool& flagRePack,
                                      const int& planeIdx, const int& mtlWidth, const int& mtlHeight,
                                      NS_StreamFile::StreamFile& streamInfos,
                                      const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
                                      NS_ImgWriter::ImgWriter& imgWriter, const NS_ImgWriter::ImgProcessTypeEnum& writerImgProcessType,
                                      const bool& flagTopOneAssignedMtl, const int& nExtrudePixels,
                                      const std::string& fileMtlBin, const bool& newFile, const std::string& rawDir,
                                      const int& globalDebugFlag, const std::string& prompt);
    void WriteObjMtlFilesCompactMtl(const bool& flagRePack, const std::string& objFile, const std::string& mtlFile,
                                    const bool& newFile, const std::string& mtlIndex,
                                    const int& cumFaceCnt, const int& cumVertexCnt, const int& cumTexCnt,
                                    const bool& flagTopOneAssignedMtl, const bool& flagDebugMeshShape,
                                    const std::string& fileFaceCamPairs, const std::string& fileMeshFaceID2ObjFaceID,
                                    const NS_MeshData::OurMeshData::OurMeshDataStructure& data);
    
    void GenNonCompactMaterialImage(NS_StreamFile::StreamFile& streamInfos,
                                     const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
                                     NS_ImgWriter::ImgWriter& imgWriter, const NS_ImgWriter::ImgProcessTypeEnum& writerImgProcessType,
                                     std::vector<bool>& flagProcessedStreamImg,
                                     const std::string& outDir,
                                     const int& globalDebugFlag, const std::string& prompt);
    void WriteObjMtlFilesNonCompactMtl(const std::string& groudIdx, const std::string& objFile,
                                     const std::string& mtlFile, const bool& newFile,
                                     std::vector<bool>& flagProcessedStreamImg,
                                     const int& cumVertexCnt, const int& cumTexCnt, int& curTexCnt,
                                     const NS_MeshData::OurMeshData::OurMeshDataStructure& data);
    
private:
    TexAlignerDataStructure texAlignerData;

    void ComputeFaceVertexVisibility(const NS_StreamFile::StreamFile& streamInfos,
                                     const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
                                     const int& globalDebugFlag, const std::string& prompt);
    void ComputeFaceCenterVisibility(const NS_StreamFile::StreamFile& streamInfos,
                                     const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
                                     const int& globalDebugFlag, const std::string& prompt);
    void ComputeFaceCenterPositiveZ(const NS_StreamFile::StreamFile& streamInfos,
                                    const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
                                    const int& globalDebugFlag, const std::string& prompt);
    void ComputeFaceArea(const NS_StreamFile::StreamFile& streamInfos,
                         const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
                         const bool& flagFaceAreaNDC,
                         const int& globalDebugFlag, const std::string& prompt);
    void ComputeFaceCameraDist(const NS_StreamFile::StreamFile& streamInfos,
                               const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
                               const int& globalDebugFlag, const std::string& prompt);
    void ComputeFaceVertexPerceptionConsistency(NS_StreamFile::StreamFile& streamInfos,
                                                const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
                                                const int& globalDebugFlag, const std::string& prompt);

    void RetrieveVertexColor(NS_StreamFile::StreamFile& streamInfos,
                            const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
                            const int& globalDebugFlag, const std::string& prompt);
    void ComputerFaceCenterNDC(Eigen::MatrixXf& faceCenterNDC, const Eigen::Vector4f& center3DCoord,
                               const NS_StreamFile::StreamFile& streamInfos);
    void ComputeValidMatrix(const int& modelIdx, Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& validMatrix);
    void ComputeScoreMatrix(const int& modelIdx, const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& validMatrix, Eigen::MatrixXf& scoreMatrix,
                            const float& unaryPotentialDummy, const float& penaltyFaceArea,
                            const float& penaltyFaceCamDist, const float& penaltyFaceVertexPerceptionConsist);
    
    void AlignTextureToMeshArgMax(const Eigen::MatrixXf& scoreMatrix, std::vector<int>& textureForTriangles);
    void AlignTextureToMeshMP(const int& modelIdx, const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& validMatrix, const Eigen::MatrixXf& scoreMatrix,
                              std::vector<int>& textureForTriangles,
                              NS_MPSolver::MPSolver<float, int>& MP, const int& nItersMP,
                              const float& pairwisePotential, const float& potentialPairOffDiagScaleDepth, const float& potentialPairOffDiagScalePercept,
                              const bool& flagSaveFullBeliefs, const std::string& fileAlBeliefs,
                              const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
                              const int& globalDebugFlag, const std::string& prompt);
    void CreatePairwisePotential(const int& modelIdx, const int& f1, const int& f2, const int& nViews,
                                 const float& potentialPairVal, const float& potentialPairOffDiagScaleDepth,
                                 const float& potentialPairOffDiagScalePercept, std::vector<float>& potentialPairVec,
                                 const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& validMatrix,
                                 const NS_MeshData::OurMeshData::OurMeshDataStructure& data);
    
    // for storing information
    void WriteNDC(const std::string& fileNDC, const bool& newFile);
    // void WriteMatrixBin(const Eigen::MatrixXf& Mat, std::ofstream& outFileStream, const bool& orderC);
    void GenRankedFaceCamPairs(const std::vector<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>>& vecValidMatrix,
                               const std::vector<Eigen::MatrixXf>& vecScoreMatrix,
                               const std::string& fileFaceCamPairs, const bool& newFile,
                               const NS_MeshData::OurMeshData::OurMeshDataStructure& data,
                               const int& globalDebugFlag, const std::string& prompt);
    
    void ObjFileWriter(const bool& flagRePack, std::ofstream& outf, std::ofstream& fcPairStream,
                       const int& cumFaceCnt, const int& cumVertexCnt, const int& cumTexCnt, nlohmann::json& jsonFaceID,
                       const std::string& mtlIndex, const bool& flagTopOneAssignedMtl, const bool& flagDebugMeshShape,
                       const NS_MeshData::OurMeshData::OurMeshDataStructure& data);

    void RegulateTextureCoords(const int& k, const int& f, const float& onePixelShiftHeight, const float& onePixelShiftWidth,
                               const std::vector<int>& dataUVCnts, std::vector<NS_TriangleR::CoordFloat*>& tmpVecMtlCoord,
                               const NS_MeshData::OurMeshData::OurMeshDataStructure& data);
};

} //end namespace

#endif /* ifndef H_TEXTURE_ALIGNER */