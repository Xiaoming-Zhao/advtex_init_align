#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <cmath>
#include <glob.h>
#include <cerrno>

#include "Utils.h"
#include "OurIO.h"
#include "StreamFile.h"
#include "OurMeshData.h"
#include "TextureAlignment.h"
#include "MPSolver.h"
#include "ImgWriter.h"

int main(int argc, char** argv)
{

    auto global_start = std::chrono::high_resolution_clock::now();

    std::cout << "Command: ";
    for (char **arg = argv; *arg; ++arg) std::cout << std::string(*arg) << " ";
    std::cout << std::endl;

    // general configuration
    int globalDebugFlag = 0;
    bool flagDebugMeshShape = false;
    std::string dataDir = ".";
    int nWorkers = 10;
    int streamTypeIndicator = 0;     // 0: unknown; 1: apple; 2: colmap
    bool flagMimicMatlab = false;
    bool flagConformalMap = true;
    bool flagCompactMtl = true;
    bool flagLoadTexUVs = false;
    int nExtrudePixels = 0;
    // store information for later DL processing
    bool flagTopOneAssignedMtl = false;
    bool flagSaveMtl = true;
    std::string fileFaceCamPairs = ".";
    // std::string fileNDC = ".";
    std::string fileCamMat = ".";
    // mesh processing
    int nPreprocSplitMesh = 0;     // 0: no split, any int > 0: number of splitted meshes
    bool flagRemesh = false;
    double maxEdgeLenDivider = 25.0;
    int iterSubdiv = 0;
    int binPackTypeIndicator = 3;     // 1: fixed_num; 2: fixed_scale; 3: sorted_then_fix_num
    int nBinPackAreasPerPlate = 1000;
    float binPackAdaptThreshold = 0.8;
    // image related
    bool flagInterpolate = false, flagAverage = false;
    int mtlWidth = 256, mtlHeight = 256;
    // float extrudeAbsTol = 0.0f;   // this controls the absolute tolerance for extrusion for each triangle on material plate
    // message passing related
    bool flagSaveAllBeliefs = false;
    int nItersMP = 10;
    int alignArgMax = 0;
    bool flagFaceAreaNDC = false;
    float unaryPotentialDummy = -2000.0f;                   // -2000 for apple stream, -200000 for T2
    float penaltyFaceArea = 1e-4;                           // apple faceArea max value ~0.01 (NDC), ~500 (pixels); penalty = 5.0f if use flagFaceAreaNDC True;
    float penaltyFaceCamDist = -100.0f;                      // 100 for apple, 1000 for T2, apple max val ~0.3
    float penaltyFaceVertexPerceptionConsist = -20.0f;       // apple max val ~0.5
    float pairwisePotentialMP = 10.0f;
    float pairwisePotentialOffDiagScaleDepth = -10.0f, pairwisePotentialOffDiagScalePercept = -10.0f;

    int cmdArgPos = -1;
    // general configuration
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--debug", argc, argv)) > 0) globalDebugFlag = std::stoi(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--debug_mesh_shape", argc, argv)) > 0) flagDebugMeshShape = (bool)std::stoi(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--data_dir", argc, argv)) > 0) dataDir = std::string(argv[cmdArgPos + 1]);
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--stream_type", argc, argv)) > 0) streamTypeIndicator = std::stoi(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--n_workers", argc, argv)) > 0) nWorkers = std::stoi(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--conformal_map", argc, argv)) > 0) flagConformalMap = (bool)std::stoi(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--compact_mtl", argc, argv)) > 0) flagCompactMtl = (bool)std::stoi(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--load_tex_uv", argc, argv)) > 0) flagLoadTexUVs = (bool)std::stoi(std::string(argv[cmdArgPos + 1]));
    // store information for later DL processing
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--top_one_mtl", argc, argv)) > 0) flagTopOneAssignedMtl = (bool)std::stoi(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--save_mtl", argc, argv)) > 0) flagSaveMtl = (bool)std::stoi(std::string(argv[cmdArgPos + 1]));
    // mesh processing
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--preprocess_split_mesh", argc, argv)) > 0) nPreprocSplitMesh = std::stoi(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--iter_subdiv", argc, argv)) > 0) iterSubdiv = std::stoi(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--bin_pack_type", argc, argv)) > 0) binPackTypeIndicator = std::stoi(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--n_areas_per_plate_bin_pack", argc, argv)) > 0) nBinPackAreasPerPlate = std::stoi(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--bin_pack_adapt_threshold", argc, argv)) > 0) binPackAdaptThreshold = std::stof(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--remesh", argc, argv)) > 0) flagRemesh = (bool)std::stoi(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--remesh_max_edge_divider", argc, argv)) > 0) maxEdgeLenDivider = std::stod(std::string(argv[cmdArgPos + 1]));
    // message passing related
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--align_arg_max", argc, argv)) > 0) alignArgMax = std::stoi(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--face_area_in_ndc", argc, argv)) > 0) flagFaceAreaNDC = (bool)std::stoi(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--save_all_beliefs", argc, argv)) > 0) flagSaveAllBeliefs = (bool)std::stoi(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--n_iter_mp", argc, argv)) > 0) nItersMP = std::stoi(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--unary_potential_dummy", argc, argv)) > 0) unaryPotentialDummy = std::stof(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--penalty_face_area", argc, argv)) > 0) penaltyFaceArea = std::stof(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--penalty_face_cam_dist", argc, argv)) > 0) penaltyFaceCamDist = std::stof(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--penalty_face_v_perception_consist", argc, argv)) > 0) penaltyFaceVertexPerceptionConsist = std::stof(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--pair_potential_mp", argc, argv)) > 0) pairwisePotentialMP = std::stof(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--pair_potential_off_diag_scale_depth", argc, argv)) > 0) pairwisePotentialOffDiagScaleDepth = std::stof(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--pair_potential_off_diag_scale_percept", argc, argv)) > 0) pairwisePotentialOffDiagScalePercept = std::stof(std::string(argv[cmdArgPos + 1]));
    // image related
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--mtl_width", argc, argv)) > 0) mtlWidth = std::stoi(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--mtl_height", argc, argv)) > 0) mtlHeight = std::stoi(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--n_extrude_pixels", argc, argv)) > 0) nExtrudePixels = std::stoi(std::string(argv[cmdArgPos + 1]));
    // if ((cmdArgPos = NS_Utils::ArgPos((char*)"--extrude_abs_tol", argc, argv)) > 0) extrudeAbsTol = std::stof(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--interpolate", argc, argv)) > 0) flagInterpolate = (bool)std::stoi(std::string(argv[cmdArgPos + 1]));
    if ((cmdArgPos = NS_Utils::ArgPos((char*)"--average", argc, argv)) > 0) flagAverage = (bool)std::stoi(std::string(argv[cmdArgPos + 1]));

    NS_MeshData::BinPackTypeEnum binPackType;
    switch(binPackTypeIndicator) {
        case 1:
            binPackType = NS_MeshData::BinPackTypeEnum::FIX_NUM;
            break;
        case 2:
            binPackType = NS_MeshData::BinPackTypeEnum::FIX_SCALE;
            break;
        case 3:
            binPackType = NS_MeshData::BinPackTypeEnum::SORTED_FIX_NUM;
            break;
        case 4:
            binPackType = NS_MeshData::BinPackTypeEnum::FACE_AREA_ADAPTIVE;
            break;
    }

    NS_StreamFile::StreamTypeEnum streamtype;
    switch(streamTypeIndicator) {
        case 0:
            streamtype = NS_StreamFile::StreamTypeEnum::STREAM_UNKNOWN;
            break;
        case 1:
            streamtype = NS_StreamFile::StreamTypeEnum::STREAM_APPLE;
            break;
        case 2:
            streamtype = NS_StreamFile::StreamTypeEnum::STREAM_COLMAP;
            break;
    }

    if (dataDir.back() == std::string("/").back()) {
        dataDir = dataDir.substr(0, dataDir.length() - 1);
    }

    flagMimicMatlab = (!flagConformalMap) && (!flagCompactMtl);

    // at most one of splitting mesh and mimicing matlab can be applied
    assert (!((bool)nPreprocSplitMesh && flagMimicMatlab));
    // at most one of debugging mesh shape and mimicing matlab can be applied
    assert (!(flagDebugMeshShape && flagMimicMatlab));
    // at most one of non-top-one mtl and mimicing matlab can be applied
    assert (!(flagTopOneAssignedMtl && flagMimicMatlab));
    // debug mesh shape must co-occur with
    // - top-one assignment to avoid saving binary files
    // - non-conformal mapping to manually add vertices and faces info
    if (flagDebugMeshShape) assert (!flagConformalMap && flagTopOneAssignedMtl);
    // No need for extrude when using conformal mapping
    if (flagConformalMap) assert(nExtrudePixels == 0);

    std::cout << std::endl << "Unary potential: " << std::endl;
    std::cout << "Dummy unary potential: " << unaryPotentialDummy << std::endl;
    std::cout << "Face area penalty: " << penaltyFaceArea << std::endl;
    std::cout << "Face camera distance penalty: " << penaltyFaceCamDist << std::endl;
    std::cout << "Face vertex's perception consistency penalty: " << penaltyFaceVertexPerceptionConsist << std::endl;
    if (alignArgMax == 0) {
        std::cout << "Pairwise potential: " << pairwisePotentialMP \
                  << "; off-diagonal scale: depth " << pairwisePotentialOffDiagScaleDepth << ", color: " << pairwisePotentialOffDiagScalePercept << std::endl;
    }
    std::cout << std::endl;

    std::string outDir;
    if (alignArgMax == 0) {
        if ((pairwisePotentialOffDiagScaleDepth == 0) && (pairwisePotentialOffDiagScalePercept == 0)) outDir = dataDir + "/output_obj_mp_only_adj";
        else if (pairwisePotentialOffDiagScaleDepth == 0) outDir = dataDir + "/output_obj_mp_only_perception";
        else if (pairwisePotentialOffDiagScalePercept == 0) outDir = dataDir + "/output_obj_mp_only_depth";
        else outDir = dataDir + "/output_obj_mp";
    } else {
        if ((penaltyFaceArea == 0) && (penaltyFaceCamDist == 0) && (penaltyFaceVertexPerceptionConsist != 0)) outDir = dataDir + "/output_obj_argmax_only_perception";
        else if ((penaltyFaceArea == 0) && (penaltyFaceCamDist != 0) && (penaltyFaceVertexPerceptionConsist == 0)) outDir = dataDir + "/output_obj_argmax_only_depth";
        else if ((penaltyFaceArea != 0) && (penaltyFaceCamDist == 0) && (penaltyFaceVertexPerceptionConsist == 0)) outDir = dataDir + "/output_obj_argmax_only_area";
        else outDir = dataDir + "/output_obj_argmax";
    }
    if (flagInterpolate) outDir += "_interpolate";
    if (flagMimicMatlab) outDir += "_matlab";
    if (flagSaveMtl) {
        outDir += "_" + std::to_string(mtlHeight) + "_" + std::to_string(mtlWidth);
        if (binPackTypeIndicator == 4) outDir += "_adapt_" + std::to_string(binPackAdaptThreshold);
        else outDir += "_" + std::to_string(nBinPackAreasPerPlate);
    } else {
        outDir += "_no_mtl";
    }
    NS_Utils::MakeDir(outDir);

    std::string fileAllBeliefs = outDir + "/all_beliefs.bin";

    // logging hyperparameters
    std::ofstream logFile(outDir + "/run.log", std::ios_base::out);

    logFile << "streamTypeIndicator: " << streamTypeIndicator << std::endl;
    logFile << "flagMimicMatlab: " << flagMimicMatlab << std::endl;
    logFile << "flagConformalMap: " << flagConformalMap << std::endl;
    logFile << "flagCompactMtl: " << flagCompactMtl << std::endl;
    logFile << "nExtrudePixels: " << nExtrudePixels << std::endl;
    // store information for later DL processing
    logFile << "flagTopOneAssignedMtl: " <<  flagTopOneAssignedMtl << std::endl;
    // mesh processing
    logFile << "nPreprocSplitMesh: " << nPreprocSplitMesh << std::endl;     // 0: no split, any int > 0: number of splitted meshes
    logFile << "flagRemesh: " << flagRemesh << std::endl;
    logFile << "maxEdgeLenDivider: " << maxEdgeLenDivider << std::endl;
    logFile << "iterSubdiv: " << iterSubdiv << std::endl;
    logFile << "binPackTypeIndicator: " << binPackTypeIndicator << std::endl;     // 1: fixed_num; 2: fixed_scale; 3: sorted_then_fix_num
    logFile << "nBinPackAreasPerPlate: " << nBinPackAreasPerPlate << std::endl;
    logFile << "binPackAdaptThreshold: " << binPackAdaptThreshold << std::endl;
    // image related
    logFile << "flagInterpolate: " << flagInterpolate << std::endl;
    logFile << "flagAverage: " << flagAverage << std::endl;
    logFile << "mtlWidth: " << mtlWidth << std::endl;
    logFile << "mtlHeight: " << mtlHeight << std::endl;
    // float extrudeAbsTol = 0.0f;   // this controls the absolute tolerance for extrusion for each triangle on material plate
    // message passing related
    logFile << "nItersMP: " << nItersMP << std::endl;
    logFile << "flagFaceAreaNDC: " << flagFaceAreaNDC << std::endl;
    logFile << "alignArgMax: " << alignArgMax << std::endl;
    logFile << "unaryPotentialDummy: " << unaryPotentialDummy << std::endl;
    logFile << "penaltyFaceArea: " << penaltyFaceArea << std::endl;
    logFile << "penaltyFaceCamDist: " << penaltyFaceCamDist << std::endl;
    logFile << "penaltyFaceVertexPerceptionConsist: " << penaltyFaceVertexPerceptionConsist << std::endl;
    logFile << "pairwisePotentialMP: " << pairwisePotentialMP << std::endl;
    logFile << "pairwisePotentialOffDiagScaleDepth: " << pairwisePotentialOffDiagScaleDepth << std::endl;
    logFile << "pairwisePotentialOffDiagScalePercept: " << pairwisePotentialOffDiagScalePercept << std::endl;

    logFile.close();

    std::string rawDir = ".";
    std::string binDir = "./";
    if (!flagTopOneAssignedMtl) {
        binDir = outDir + "/bin";
        NS_Utils::MakeDir(binDir);
        if (!flagMimicMatlab) assert(flagCompactMtl);
        fileFaceCamPairs = binDir + "/face_cam_pairs.bin";
        // fileNDC = binDir + "/ndc.bin";
        fileCamMat = binDir + "/camera_mats.bin";
        // write raw images into disk
        rawDir = outDir + "/raw";
        NS_Utils::MakeDir(rawDir);
    }

    // NOTE: C++17 has en elegant std::filesystem. Not sure whether we want to move to C++17.
    std::vector<int> allFileIndices;

    glob_t globbuf;
    std::string search_pattern = dataDir + "/Faces.*";
    int err = glob(search_pattern.c_str(), 0, NULL, &globbuf);
    if (err == 0) {
        // max_file_index = globbuf.gl_pathc;
        for (std::size_t i = 0; i < globbuf.gl_pathc; i++) {
            std::cout << "Find " << globbuf.gl_pathv[i] << std::endl;
            std::string tmp_face_f(globbuf.gl_pathv[i]);
            std::size_t tmpDotPos = tmp_face_f.find_last_of(".");
            allFileIndices.push_back(std::stoi(tmp_face_f.substr(tmpDotPos + 1)));
        }
        globfree(&globbuf);
    } else {
        std::cerr << "Unable to find suitable faces and vertices data." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::sort(allFileIndices.begin(), allFileIndices.end());
    std::cout << "Find " << allFileIndices.size() << " Faces/Vertices files" << std::endl;

    if (nPreprocSplitMesh > 0) {
        std::string splitMeshDir = dataDir + "/splitted_mesh_" + std::to_string(nPreprocSplitMesh);
        NS_Utils::MakeDir(splitMeshDir);
        NS_MeshData::OurMeshData OurMesh;
        OurMesh.PartitionMeshData(dataDir, allFileIndices, splitMeshDir, nPreprocSplitMesh);
        return 0;
    }

    std::string stream_f = dataDir + "/Recv.stream";
    NS_StreamFile::StreamFile OurStream;
    OurStream.IndexFile(stream_f, globalDebugFlag, streamtype);

    std::vector<bool> flagProcessedStreamImg;
    if (!flagCompactMtl) {
        // globally track which stream images are utilized
        flagProcessedStreamImg.resize(OurStream.NumberOfEntries(), false);
    }

    if (flagSaveAllBeliefs) {
        std::ofstream outf(fileAllBeliefs, std::ios_base::binary | std::ios_base::out);
        if (!outf.is_open()) {
            std::cout << "Unable to open binary file for saving all beliefs." << std::endl;
            exit(EXIT_FAILURE);
        }
        int nViews = OurStream.NumberOfEntries();
        outf.write((const char*)&nViews, sizeof(nViews));
        outf.close();
    }
    
    if (!flagDebugMeshShape) {
        OurStream.ReadAllRGBImages();
        OurStream.ReadAllDepthMaps();
        if (!flagTopOneAssignedMtl) {
            std::string binDepth = binDir + "/depth.bin";
            OurStream.SaveDepthMapToDisk(binDepth);
        }
    }

    if (!flagTopOneAssignedMtl) OurStream.WriteCameraMatrixBinary(fileCamMat);

    NS_ImgWriter::ImgWriter imgWriter(nWorkers);

    unsigned int cumVertexCnt = 0, cumTexCnt = 0, cumFaceCnt = 0;
    std::string objFile = outDir + "/TexAlign.obj";
    std::string mtlFile = outDir + "/TexAlign.mtl";
    std::string fileMeshFaceID2ObjFaceID = outDir + "/mesh_face_id_to_obj_face_id.json";

    for (int fileCnt = 0; fileCnt < allFileIndices.size(); fileCnt++) {

        unsigned int fileIndex = allFileIndices[fileCnt];

        std::cout << std::endl << "Start processing " \
                  << dataDir + "/Vertices." + std::to_string(fileIndex) \
                  << " and " << dataDir + "/Faces." + std::to_string(fileIndex) << "..." << std::endl;

        std::string logPrompt = "[" + std::to_string(fileCnt + 1) + \
                                  "(idx " + std::to_string(fileIndex) + ")/" + std::to_string(allFileIndices.size()) + "] ";

        auto file_start = std::chrono::high_resolution_clock::now();
    
        std::string vf(dataDir + "/Vertices." + std::to_string(fileIndex));
        std::string ff(dataDir + "/Faces." + std::to_string(fileIndex));
        std::string texf(dataDir + "/TexVertices." + std::to_string(fileIndex));

        NS_MeshData::OurMeshData OurMesh;
        if (flagLoadTexUVs) {
            OurMesh.CreateMeshFromDataNoPostProcess(vf, ff, texf, globalDebugFlag, logPrompt);
        } else {
            OurMesh.CreateMeshFromData(vf, ff, flagConformalMap, globalDebugFlag, logPrompt);
        }

        if (globalDebugFlag == 0) {
            std::cout << logPrompt + "Complete reading faces and vertices " << fileIndex << std::endl;
        }

        NS_MPSolver::MPSolver<float, int> MP;

        bool flagRePack = true;
        // if (flagMimicMatlab || flagDebugMeshShape) flagRePack = false;
        if (flagMimicMatlab || flagLoadTexUVs) flagRePack = false;

        if (flagDebugMeshShape || !flagSaveMtl) {
            OurMesh.CreatePlaceholderInfo();
        }
        else
        {
            if (flagConformalMap && !flagLoadTexUVs)
            {
                if (streamtype == NS_StreamFile::STREAM_COLMAP) {
                    OurMesh.BffPreProcessing(flagRemesh, maxEdgeLenDivider, globalDebugFlag, logPrompt);
                }
                
                OurMesh.SubdivideMesh(iterSubdiv, globalDebugFlag, logPrompt);
    
                OurMesh.FlattenMesh(globalDebugFlag, logPrompt);
                
                OurMesh.TextureTriOverlapCheck(logPrompt);
    
                // bool flagPlaceholder = false;
                // if (flagDebugMeshShape) flagPlaceholder = true;
                // OurMesh.ResolveOverlappedConformalMap(MP, nItersMP, globalDebugFlag, logPrompt);
                OurMesh.ResolveOverlappedConformalMap(MP, 10, globalDebugFlag, logPrompt);
            }

            if ((flagLoadTexUVs || !flagConformalMap) && flagCompactMtl && alignArgMax == 0) OurMesh.ComputeTriangleAdjacency3D(globalDebugFlag, logPrompt);

        }
        
        const NS_MeshData::OurMeshData::OurMeshDataStructure& preBinPackMeshData = OurMesh.GetData();

        
        NS_TextureAlign::TextureAligner textureAlign3D; 
        
        if (!flagDebugMeshShape) {
            textureAlign3D.ComputeNDC(OurStream, preBinPackMeshData, globalDebugFlag, logPrompt);
            textureAlign3D.ComputeCues(OurStream, preBinPackMeshData, flagFaceAreaNDC, globalDebugFlag, logPrompt);

            if (flagCompactMtl && flagSaveMtl && !flagLoadTexUVs)
            {
                const Eigen::MatrixXf& NDC = textureAlign3D.GetNDC();
                const std::vector<Eigen::MatrixXf> faceAreaMatrix = textureAlign3D.GetValidFaceAreaMatrix();
                // const std::vector<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>>& validMatrix = textureAlign3D.GetValidMatrix();
                OurMesh.BinPackTextureTriangles2D(binPackType, flagConformalMap, nBinPackAreasPerPlate, OurStream, NDC, faceAreaMatrix,
                                                  mtlHeight, mtlWidth, binPackAdaptThreshold, globalDebugFlag, logPrompt);
            }
        }

        const NS_MeshData::OurMeshData::OurMeshDataStructure& wholeMeshData = OurMesh.GetData();

        if (!flagDebugMeshShape) {

            if (!flagTopOneAssignedMtl) textureAlign3D.WriteBin(binDir, fileCnt == 0, globalDebugFlag, logPrompt);
    
            enum NS_TextureAlign::TextureAligner::TextureAlignmentApproach alignApproach;
            if (alignArgMax == 0) {
                alignApproach = textureAlign3D.TextureAlignmentApproach::MessagePassing;
            } else {
                alignApproach = textureAlign3D.TextureAlignmentApproach::ArgMax;
            }
    
            if (flagMimicMatlab) assert (alignArgMax == 1);
            
            textureAlign3D.AlignTextureToMesh(alignApproach,
                                              unaryPotentialDummy, penaltyFaceArea,
                                              penaltyFaceCamDist, penaltyFaceVertexPerceptionConsist,
                                              MP, nItersMP, pairwisePotentialMP,
                                              pairwisePotentialOffDiagScaleDepth, pairwisePotentialOffDiagScalePercept,
                                              flagSaveAllBeliefs, fileAllBeliefs, wholeMeshData,
                                              flagTopOneAssignedMtl, fileFaceCamPairs, fileCnt == 0,
                                              globalDebugFlag, logPrompt);
        }

        int curTexCnt = 0;
        if (!flagCompactMtl) {
              textureAlign3D.GenNonCompactMaterialImage(OurStream, wholeMeshData, imgWriter, NS_ImgWriter::ImgProcessTypeEnum::ROT90CCW,
                                                        flagProcessedStreamImg, outDir, globalDebugFlag, logPrompt);
              textureAlign3D.WriteObjMtlFilesNonCompactMtl(std::to_string(fileIndex), objFile, mtlFile, fileCnt == 0, flagProcessedStreamImg,
                                                         cumVertexCnt, cumTexCnt, curTexCnt, wholeMeshData);
        } else {
            for (int planeIdx = 0; planeIdx <= wholeMeshData.maxPlaneIdx; planeIdx++) {
                std::string mtlIdx = std::to_string(fileIndex) + "_" + std::to_string(planeIdx);
                if (!flagDebugMeshShape) {
                    if (flagTopOneAssignedMtl)
                    {
                        std::string fileMtl = outDir + "/mtl_" + mtlIdx + ".png";
                        std::string fileMtlCover = outDir + "/mtl_mask_" + mtlIdx + ".png";
                        textureAlign3D.GenCompactMaterialImage(flagRePack, flagInterpolate, flagAverage, planeIdx, mtlWidth, mtlHeight,
                                                               OurStream, wholeMeshData,
                                                               imgWriter, NS_ImgWriter::ImgProcessTypeEnum::TRANSPOSE_ROT90CCW,
                                                               flagConformalMap, nExtrudePixels, fileMtl, fileMtlCover,
                                                               globalDebugFlag, logPrompt);
                    }
                    else {
                        if (flagSaveMtl) {
                            std::string fileMtl = binDir + "/mtl_" + mtlIdx + ".bin";
                            std::string fileMtlCover = binDir + "/mtl_mask_" + mtlIdx + ".bin";
                            textureAlign3D.GenCompactMaterialBinaryFile(flagRePack, planeIdx, mtlWidth, mtlHeight,
                                                                        OurStream, wholeMeshData,
                                                                        imgWriter, NS_ImgWriter::ImgProcessTypeEnum::TRANSPOSE_ROT90CCW,
                                                                        flagTopOneAssignedMtl, nExtrudePixels, fileMtl, fileCnt == 0, rawDir,
                                                                        globalDebugFlag, logPrompt);
                        }

                        if ((fileCnt == 0) && (planeIdx == 0)) {
                            OurStream.SaveRGBToDisk(imgWriter, rawDir);
                            // textureAlign3D.WriteRawRGB(OurStream, imgWriter, rawDir);
                        }
                    }
                }
            }
    
            textureAlign3D.WriteObjMtlFilesCompactMtl(flagRePack, objFile, mtlFile, fileCnt == 0, std::to_string(fileIndex),
                                                      cumFaceCnt, cumVertexCnt, cumTexCnt, flagTopOneAssignedMtl, flagDebugMeshShape || !flagSaveMtl,
                                                      fileFaceCamPairs, fileMeshFaceID2ObjFaceID, wholeMeshData);
        }

        // update global vertex and texture count
        cumFaceCnt += wholeMeshData.nPointTriangles;
        cumVertexCnt += wholeMeshData.Points.cols();
        if (!flagCompactMtl) {
            cumTexCnt += curTexCnt;
        } else {
            if (flagRePack) {
                for (auto it: wholeMeshData.uvPtsAfterRePack) cumTexCnt += it.size();
            } else {
                for (auto it: wholeMeshData.uvPts) cumTexCnt += it.size();
            }
        }

        auto file_finish = std::chrono::high_resolution_clock::now();
        std::cout << logPrompt + "... complete processing " \
                  << dataDir + "/Vertices." + std::to_string(fileIndex) \
                  << " and " << dataDir + "/Faces." + std::to_string(fileIndex) << ": " \
                  << std::chrono::duration_cast<std::chrono::seconds>(file_finish - file_start).count() << "s" << std::endl;
    }

    imgWriter.WaitUntilFinish();

    auto global_finish = std::chrono::high_resolution_clock::now();
    std::cout << std::endl << "Complete processing " << dataDir << " in " \
              << std::chrono::duration_cast<std::chrono::seconds>(global_finish - global_start).count() << " s" << std::endl;

    return 0;
}