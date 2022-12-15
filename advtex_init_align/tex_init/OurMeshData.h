#ifndef H_OURMESH_DATA
#define H_OURMESH_DATA

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "StreamFile.h"
#include "Bff.h"
#include "MPSolver.h"
#include "Utils.h"

typedef CGAL::Simple_cartesian<double>              Kernel;
typedef Kernel::Point_3                             Point_3;
//typedef Kernel::Point_2                             Point_2;
typedef CGAL::Surface_mesh<Point_3>                 SurfaceMesh;

#define N_BIN_PACK_AREAS_PER_PLATE 1000   // 500 for conformal mapping; 1000 for non-conformal mapping
#define BIN_PACK_BOX_LENGTH 100000
#define BIN_PACK_UNITS_PER_INT 1.0e-5

namespace NS_MeshData {

enum BinPackTypeEnum {FIX_NUM, FIX_SCALE, SORTED_FIX_NUM, FACE_AREA_ADAPTIVE};

typedef Eigen::Matrix<double, 3, Eigen::Dynamic> Matrix3Xd;

class OurMeshData {
public:

    struct OurMeshDataStructure {
        // 3D points
        Matrix3Xd Points;   // (3, #points)

        // 3D triangles
        int nPointTriangles = 0;
        std::vector<std::vector<NS_Utils::Triple>> pointTriangles;

        // 2D Texture, points
        std::vector<std::vector<Eigen::Vector2d>> uvPts;
        std::vector<int> uvCnts;

        // 2D Texture triangles
        int nTextureTriangles = 0;
        std::vector<std::vector<NS_Utils::Triple>> textureTriangles;
        
        std::vector<std::vector<int>> triangleGlobalIdx;
        std::vector<std::pair<unsigned int, unsigned int>> triangleGlobalIdxToPairIdx;

        // adjacency
        int nMeshTriangleAdjacency3D = 0;
        std::vector<std::vector<std::set<unsigned int>>> infoMeshTriangleAdjacency3D;  // each element in std::set<unsigned int> represents faceID in corresponding modelID
        int nTextureTriangleAdjacency2D = 0;
        std::vector<std::vector<std::set<unsigned int>>> infoTextureTriangleAdjacency2D;

        // conformal mapping overlap
        int maxTimesTextureTriangleOverlap = 0;
        int nTextureTrianglesWithOverlap = 0;    // this counts triangles, one triangle could overlap with other triangle several times, but we it will be counted only once.
        int nTextureTriangleOverlaps = 0;        // this counts overlaps.
        std::vector<unsigned int> flagsTextureTriangleOverlap;
        std::vector<std::vector<std::vector<unsigned int>>> infoTextureTriangleOverlap;   // every two elements in std::set<unsigned int> represent (modelID, faceID)
        int maxPlaneIdx = 0;  // this counts maximum plane ID after resolving overlapping with MP
        std::vector<unsigned int> planeIdxAfterMP;   // this contains plane assignment after message passing
        std::vector<unsigned int> nTextureTrianglesOnPlane;

        // for resolving conformal map's overlap 
        std::vector<std::vector<unsigned int>> infoTextureTriangleAreaIdx;   // assign an area index to each triangle
        std::vector<std::vector<NS_Utils::Tuple>> infoAreaTriangleIdxSet;  // each element is a set of triangle's indices (k, f), which form a connected area

        // bin-pack texture triangles
        std::vector<std::vector<Eigen::Vector2d>> uvPtsAfterRePack;
        std::vector<unsigned int> planeIdxAfterRePack;
        std::vector<int> uvCntsAfterRePack;

        void resize(int nVertices, int modelSize) {
            Points.conservativeResize(3, nVertices);
            uvPts.resize(modelSize);
            pointTriangles.resize(modelSize);     // global index
            textureTriangles.resize(modelSize);   // global index
        }
    };
private:
    SurfaceMesh sm;
    OurMeshDataStructure data;

    void SimplifyMesh();
    void Edge_Analysis(float& minEdgeLen, float& maxEdgeLen, float& meanEdgeLen);
    void CleanMesh();
    int CountNonManifoldVertices();

    void ComputeTriangleAdjacency3DInBFF(bff::Model& model, const int& globalDebugFlag, const std::string& prompt);
    void ComputeTriangleAdjacency2D(const int& globalDebugFlag, const std::string& prompt);
    // bool CheckAdjacency2D(const int& modelIdx, const int& faceIdx1, const int& faceIdx2, const int& nEdgeCombine, const int (&edgeCombination)[18][4]);

    // Interaction between CGAL and BFF
    bool ConvertCGAL2BFF(bff::Model& model, std::string& error);
    bool convertC2B(bff::Model& model, std::string& error);

    // BFF computes conformal mapping
    void loadModel(bff::Model& model, std::vector<bool>& surfaceIsClosed, const int& globalDebugFlag, const std::string& prompt);
    void flatten(bff::Model& model, const std::vector<bool>& surfaceIsClosed, int nCones, bool flattenToDisk, bool mapToSphere);
    void convertBFFMesh(bff::Model& model, const std::vector<bool>& surfaceIsClosed, bool mapToSphere, bool normalize);
    Eigen::Vector2d getUV(bff::Vector uv, bool mapToSphere, double sphereRadius, double meshRadius,
                         const bff::Vector& oldCenter, const bff::Vector& newCenter,
                         const bff::Vector& minExtent, double extent, bool flipped, bool normalize);
    void writeModelUVs(const std::string& outputPath, bff::Model& model, const std::vector<bool>& surfaceIsClosed, bool mapToSphere, bool normalizeUVs);
    void BFFPostProcessing(const int& globalDebugFlag, const std::string& prompt);

    // Packing non-overlapped areas
    void SearchConnectedArea(const int& planeIdx, const int& modelIdx, const int& faceIdx, const int& connectedAreaIdx,
                             std::vector<NS_Utils::Tuple>& idxSet, std::vector<bool>& visited,
                             std::vector<bff::Vector>& minBounds, std::vector<bff::Vector>& maxBounds);
    void GetBounds(const int& modelIdx, const int& faceIdx, bff::Vector& minBounds, bff::Vector& maxBounds);
    void BinPackTextureTriangles2DSinglePlane(const bool& flagConformalMap, const int& planeIdx, // const double& totalArea,
                                              const std::vector<int>& areaIdxs, const std::vector<double>& areas,
                                              const std::vector<bff::Vector>& minBounds, const std::vector<bff::Vector>& maxBounds,
                                              std::vector<std::vector<int>>& visitedUVPts, std::vector<unsigned int>& addUVCnts);
    // for conformal mapping
    void BinPackPreprocessForConformalMap(std::vector<double>& areas, std::vector<bff::Vector>& minBounds, std::vector<bff::Vector>& maxBounds,
                                          const int& globalDebugFlag, const std::string& prompt);
    void BinPackPostprocessForConformalMap(const int& planeIdx, const int& nConnectedAreas, const std::vector<int>& areaIdxs,
                                           const bff::Vector& minExtent, const double& extent, const std::vector<bool>& flippedBins,
                                           const std::vector<bff::Vector>& originalCenters, const std::vector<bff::Vector>& newCenters,
                                           std::vector<std::vector<int>>& visitedUVPts, std::vector<unsigned int>& addUVCnts);
    // for non-conformal mapping
    void BinPackPreprocessForNonConformalMap(std::vector<double>& areas,
                                             std::vector<bff::Vector>& minBounds, std::vector<bff::Vector>& maxBounds,
                                             const bool& flagAreaMesh, const NS_StreamFile::StreamFile& streamInfos,
                                             const Eigen::MatrixXf& NDC, const std::vector<Eigen::MatrixXf>& faceAreaMatrix,
                                             const int& globalDebugFlag, const std::string& prompt);
    void BinPackPostprocessForNonConformalMap(const int& planeIdx, const int& nConnectedAreas, const std::vector<int>& areaIdxs,
                                              const bff::Vector& minExtent, const double& extent, const std::vector<bool>& flippedBins,
                                              const std::vector<bff::Vector>& originalCenters, const std::vector<bff::Vector>& newCenters);
public:
    OurMeshData();
    ~OurMeshData();
    const OurMeshDataStructure& GetData() const;

    void CreatePlaceholderInfo();

    void PartitionMeshData(const std::string& dataDir, const std::vector<int>& allFileIndices,
                           // const std::string& vertexFile, const std::string& faceFile,
                           const std::string& outDir, const int& nParts);

    // APIs for 3D mesh data
    void CreateMeshFromData(const std::string& vertexFile, const std::string& faceFile, const bool& flagConformalMap, const int& globalDebugFlag, const std::string& prompt);
    void CreateMeshFromDataNoPostProcess(const std::string& vertexFile, const std::string& faceFile, const std::string& uvFile,
                                         const int& globalDebugFlag, const std::string& prompt);
    void CreateSubMeshFromMesh(const OurMeshDataStructure& wholeMeshData, const int& planeIdx);
    void SubdivideMesh(const int NumberOfIterations, const int& globalDebugFlag, const std::string& prompt);
    void ComputeTriangleAdjacency3D(const int& globalDebugFlag, const std::string& prompt);

    // APIs for 2D texture from BFF
    void BffPreProcessing(const bool& flagRemesh, const double& maxEdgeLenDivider, const int& globalDebugFlag, const std::string& prompt);
    void FlattenMesh(const int& globalDebugFlag, const std::string& prompt, int flagWriteUV=0);  // NOTE: flagWriteUV is used for debugging. Remove later
    void TextureTriOverlapCheck(std::string& prompt);

    // APIs for resolving overlapping
    // put an explicit instantiation here to avoid making OurMeshData class into a class template
    void ResolveOverlappedConformalMap(NS_MPSolver::MPSolver<float, int>& MP, const int& nItersMP, // const bool& flagPlaceHolder,
                                       const int& globalDebugFlag, const std::string& prompt);
    void BinPackTextureTriangles2D(const BinPackTypeEnum& binPackType, const bool& flagConformalMap, const int& nBinPackAreasPerPlate,
                                   const NS_StreamFile::StreamFile& streamInfos,
                                   const Eigen::MatrixXf& NDC, const std::vector<Eigen::MatrixXf>& faceAreaMatrix,
                                   const int& mtlHeight, const int& mtlWidth, const float& adapatThreshold,
                                   const int& globalDebugFlag, const std::string& prompt);
};

} //end namespace

#endif