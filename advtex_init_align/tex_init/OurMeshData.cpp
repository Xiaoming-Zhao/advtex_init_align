#include <iostream>
#include <vector>
#include <chrono>
#include <queue>
#include <limits>

#define CGAL_PMP_REMESHING_VERBOSE
#define CGAL_SURFACE_SIMPLIFICATION_ENABLE_TRACE 0
//#define CGAL_SURFACE_SIMPLIFICATION_ENABLE_LT_TRACE 0

#include <omp.h>
#include <CGAL/subdivision_method_3.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/merge_border_vertices.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Count_ratio_stop_predicate.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Bounded_normal_change_placement.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/GarlandHeckbert_policies.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/LindstromTurk_cost.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/LindstromTurk_placement.h>
#include <CGAL/boost/graph/partition.h>
#include <CGAL/boost/graph/METIS/partition_graph.h>

#include "MeshIO.h"
#include "BinPacking.h"
#include "ConePlacement.h"
#include "Cutter.h"
#include "HoleFiller.h"
#include "Generators.h"
#include "Rect.h"

#include "OurMeshData.h"
#include "OurIO.h"

typedef boost::graph_traits<SurfaceMesh>::vertex_descriptor     vertex_descriptor;
typedef boost::graph_traits<SurfaceMesh>::halfedge_descriptor   halfedge_descriptor;
typedef boost::graph_traits<SurfaceMesh>::face_descriptor       face_descriptor;

namespace PMP = CGAL::Polygon_mesh_processing;
namespace SMS = CGAL::Surface_mesh_simplification;

namespace NS_MeshData {

OurMeshData::OurMeshData() {}

OurMeshData::~OurMeshData() {}

const OurMeshData::OurMeshDataStructure& OurMeshData::GetData() const {
    return data;
}

void OurMeshData::PartitionMeshData(const std::string& dataDir, const std::vector<int>& allFileIndices,
                                    // const std::string& vertexFile, const std::string& faceFile,
                                    const std::string& outDir, const int& nParts) {
    SurfaceMesh rawSM;

    std::cout << "Start building mesh from " << allFileIndices.size() << " files ..." << std::endl;

    for (int fileCnt = 0; fileCnt < allFileIndices.size(); fileCnt++) {

        unsigned int fileIndex = allFileIndices[fileCnt];
        std::string vertexFile = dataDir + "/Vertices." + std::to_string(fileIndex);
        std::string faceFile = dataDir + "/Faces." + std::to_string(fileIndex);

        std::vector<float> coords;
        NS_OurIO::TextureIO::readFileIntoVector(coords, vertexFile);
    
        std::vector<unsigned int> ixs;
        NS_OurIO::TextureIO::readFileIntoVector(ixs, faceFile);

        std::vector<SurfaceMesh::Vertex_index> vix;

        for(int k = 0; k < coords.size() / 3; ++k) {
            vix.push_back(rawSM.add_vertex(Point_3((double)coords[3 * k], (double)coords[3 * k + 1], (double)coords[3 * k + 2])));
        }
    
        for(int k = 0; k < ixs.size() / 3; ++k) {
            rawSM.add_face(vix[ixs[3 * k]], vix[ixs[3 * k + 1]], vix[ixs[3 * k + 2]]);
        }
    }

    std::cout << "... done building mesh." << std::endl;

    std::cout << "Start partitioning original mesh ..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    typedef CGAL::dynamic_face_property_t<int>                                  Face_property_tag;
    typedef boost::property_map<SurfaceMesh, Face_property_tag>::type           Face_id_map;
    Face_id_map partition_id_map = get(Face_property_tag(), rawSM);

    CGAL::METIS::partition_graph(rawSM, nParts,
           CGAL::parameters::face_partition_id_map(partition_id_map)//.vertex_partition_id_map(partition_vid_map)
    );

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "... done partitioning mesh in " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms" << std::endl;

    std::cout << "Start re-indexing ..." << std::endl;
    start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<vertex_descriptor> > ptPosition(nParts, std::vector<vertex_descriptor>());
    struct triplet{int v[3];};
    std::vector<std::vector<triplet> > faces(nParts, std::vector<triplet>());
    for(auto f : rawSM.faces()) {
        int part = get(partition_id_map, f);
        //std::cout << f.idx() << "/" << rawSM.faces().size() << ": " << part << std::endl;
        halfedge_descriptor h_start = rawSM.halfedge(f);
        halfedge_descriptor h = h_start;
        int cnt = 0;
        triplet tmp;
        do {
            assert(cnt<3);
            vertex_descriptor vert = rawSM.source(h);
            std::vector<vertex_descriptor>::iterator it = std::find(ptPosition[part].begin(), ptPosition[part].end(), vert);
            int newIX = it - ptPosition[part].begin();
            if(it == ptPosition[part].end()) {
                ptPosition[part].push_back(vert);
            }
            tmp.v[cnt] = newIX;
            h = rawSM.next(h);
            ++cnt;
        } while(h!=h_start);
        assert(cnt == 3);
        faces[part].push_back(tmp);
    }

    finish = std::chrono::high_resolution_clock::now();
    std::cout << "... done re-indexing in " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms" << std::endl;

    const SurfaceMesh::Property_map<vertex_descriptor, Point_3> ptCoords = rawSM.points();
    for(int k = 0; k < ptPosition.size(); ++k) {
        std::string vfn(outDir + "/Vertices." + std::to_string(k));
        std::ofstream vofs(vfn, std::ios_base::out | std::ios_base::binary);
        float tmpV;
        for (auto v: ptPosition[k]) {
            tmpV = (float)ptCoords[v].x();
            vofs.write((const char*)&tmpV, sizeof(float));
            tmpV = (float)ptCoords[v].y();
            vofs.write((const char*)&tmpV, sizeof(float));
            tmpV = (float)ptCoords[v].z();
            vofs.write((const char*)&tmpV, sizeof(float));
        }
        vofs.close();

        std::string ffn(outDir + "/Faces." + std::to_string(k));
        std::ofstream fofs(ffn, std::ios_base::out | std::ios_base::binary);
        for (auto f: faces[k]) {
            fofs.write((const char*)&f.v[0], 3*sizeof(unsigned int));
        }
        fofs.close();
    }
}


void OurMeshData::CreateMeshFromData(const std::string& vertexFile, const std::string& faceFile, const bool& flagConformalMap,
                                     const int& globalDebugFlag, const std::string& prompt)
{
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<float> coords;
    NS_OurIO::TextureIO::readFileIntoVector(coords, vertexFile);

    std::vector<unsigned int> ixs;
    NS_OurIO::TextureIO::readFileIntoVector(ixs, faceFile);

    std::vector<SurfaceMesh::Vertex_index> vix;
    for(int k = 0; k < coords.size() / 3; ++k) {
        vix.push_back(sm.add_vertex(Point_3((double)coords[3 * k], (double)coords[3 * k + 1], (double)coords[3 * k + 2])));
    }
    
    for(int k = 0; k < ixs.size() / 3; ++k) {
        sm.add_face(vix[ixs[3 * k]], vix[ixs[3 * k + 1]], vix[ixs[3 * k + 2]]);
    }

    if (!flagConformalMap)
    {
        // triangulate mesh
        // NOTE: when using conformal mapping, BFF will triangulate mesh itself.
        for (auto fit: sm.faces()) {
            if (sm.next(sm.next(sm.halfedge(fit))) != sm.prev(sm.halfedge(fit))) {
                PMP::triangulate_faces(sm);
                break;
            }
        }

        // check again
        for (auto fit: sm.faces()) {
            assert (sm.next(sm.next(sm.halfedge(fit))) == sm.prev(sm.halfedge(fit)));
        }

        // fill Points, pointTriangles, triangleGlobalIdx
        int nVertices = sm.number_of_vertices();
        int nTriangles = sm.number_of_faces();

        data.resize(nVertices, 1);

        for(vertex_descriptor vd : sm.vertices()) {
            data.Points(0, (int)vd) = (double)sm.point(vd)[0];
            data.Points(1, (int)vd) = (double)sm.point(vd)[1];
            data.Points(2, (int)vd) = (double)sm.point(vd)[2];
        }

        std::vector<int> tmpTriangleIdx;
        for(auto fit: sm.faces()) {
            NS_Utils::Triple pT;
            int v_cnt = 0;
            for(vertex_descriptor vd : vertices_around_face(sm.halfedge(fit), sm)){
                pT.c[v_cnt] = (int)vd;
                v_cnt++;
                assert ((int)vd < nVertices);
            }
            data.pointTriangles[0].push_back(pT);
            tmpTriangleIdx.push_back((int)fit);
        }
        data.nPointTriangles = data.pointTriangles[0].size();
        data.triangleGlobalIdx.push_back(tmpTriangleIdx);
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << prompt + "complete creating mesh from data (" \
              << coords.size() / 3 << " vertices, " << ixs.size() / 3 << " faces) in " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms" << std::endl;
}


void OurMeshData::CreateMeshFromDataNoPostProcess(const std::string& vertexFile, const std::string& faceFile, const std::string& uvFile,
                                     const int& globalDebugFlag, const std::string& prompt)
{
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<float> coords;
    NS_OurIO::TextureIO::readFileIntoVector(coords, vertexFile);

    std::vector<unsigned int> ixs;
    NS_OurIO::TextureIO::readFileIntoVector(ixs, faceFile);

    std::vector<float> uvCoords;
    NS_OurIO::TextureIO::readFileIntoVector(uvCoords, uvFile);

    std::cout << "#verts: " << coords.size() / 3 << "; #tex_verts: " << uvCoords.size() / 2 << std::endl;
    assert (coords.size() / 3 == uvCoords.size() / 2);

    int nVertices = int(coords.size() / 3);
    data.resize(nVertices, 1);
    data.uvCnts.resize(1);

    for(int k = 0; k < coords.size() / 3; ++k) {
        data.Points(0, k) = (double)coords[3 * k];
        data.Points(1, k) = (double)coords[3 * k + 1];
        data.Points(2, k) = (double)coords[3 * k + 2];
        // if (k < 10) {
        //     std::cout << data.Points(0, k)  << std::endl;
        // }
    }

    for(int k = 0; k < uvCoords.size() / 2; ++k) {
        Eigen::Vector2d tmp((double)uvCoords[2 * k], (double)uvCoords[2 * k + 1]);
        data.uvPts[0].push_back(tmp);
        // if (k < 10) {
        //     std::cout << tmp << std::endl;
        // }
    }

    std::vector<int> tmpTriangleIdx;
    for(int k = 0; k < ixs.size() / 3; k++) {
        NS_Utils::Triple pT;
        for(int v_cnt = 0; v_cnt < 3; v_cnt++){
            pT.c[v_cnt] = (int)ixs[3 * k + v_cnt];
            assert (3 * k + v_cnt < nVertices);
        }
        data.pointTriangles[0].push_back(pT);
        data.textureTriangles[0].push_back(pT);
        tmpTriangleIdx.push_back(k);
    }
    data.nPointTriangles = data.pointTriangles[0].size();
    data.nTextureTriangles = data.pointTriangles[0].size();
    data.triangleGlobalIdx.push_back(tmpTriangleIdx);

    data.planeIdxAfterMP.resize(data.nPointTriangles, 0);

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << prompt + "complete creating mesh from data (" \
              << coords.size() / 3 << " vertices, " << uvCoords.size() / 2 << " UVs, " << ixs.size() / 3 << " faces) in " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms" << std::endl;
}


void OurMeshData::BffPreProcessing(const bool& flagRemesh, const double& maxEdgeLenDivider, const int& globalDebugFlag, const std::string& prompt)
{
    // std::ofstream os(vertexFile + ".off");
    // os.precision(17);
    // os << sm;
    // os.close();

    /* DEBUG start */

    if(sm.has_garbage()) {std::cout << "Collect Garbage" << std::endl; sm.collect_garbage(); std::cout << "after collecting garbage #vertices: " << sm.vertices().end() - sm.vertices().begin() << std::endl;}
    CleanMesh();
    if(sm.has_garbage()) {std::cout << "Collect Garbage" << std::endl; sm.collect_garbage(); std::cout << "after collecting garbage #vertices: " << sm.vertices().end() - sm.vertices().begin() << std::endl;}

    if (flagRemesh) {
        float minEdgeLen = 0.0f, maxEdgeLen = 0.0f, meanEdgeLen = 0.0f;
        Edge_Analysis(minEdgeLen, maxEdgeLen, meanEdgeLen);
        // PMP::isotropic_remeshing(sm.faces(), (double)maxEdgeLen / maxEdgeLenDivider, sm);
        PMP::isotropic_remeshing(sm.faces(), std::max((double)meanEdgeLen, (double)maxEdgeLen / maxEdgeLenDivider), sm);
    }

    if(sm.has_garbage()) {std::cout << "Collect Garbage" << std::endl; sm.collect_garbage(); std::cout << "after collecting garbage #vertices: " << sm.vertices().end() - sm.vertices().begin() << std::endl;}
    for(int k = 0; k < 20; ++k) {
        CleanMesh();
        if(sm.has_garbage()) {std::cout << "Collect Garbage" << std::endl; sm.collect_garbage(); std::cout << "after collecting garbage #vertices: " << sm.vertices().end() - sm.vertices().begin() << std::endl;}
    }

}

int OurMeshData::CountNonManifoldVertices() {
    int counter = 0;
    for(vertex_descriptor v : vertices(sm)) {
      if(PMP::is_non_manifold_vertex(v, sm)) {
        // std::cout << "vertex " << v << " is non-manifold" << std::endl;
        ++counter;
      }
    }
    return counter;
}

void OurMeshData::CleanMesh() {

    PMP::remove_isolated_vertices(sm);

    size_t new_verts = PMP::duplicate_non_manifold_vertices(sm);

    std::cout << "Cleaning: " << new_verts << " " << std::endl; // << v1 << " " << v2 << std::endl;
}

void OurMeshData::Edge_Analysis(float& minEdgeLen, float& maxEdgeLen, float& meanEdgeLen){
  // float meanEdgeLen = 0, minEdgeLen, maxEdgeLen;
  float length;
  int count = 0; bool init = true;
  const SurfaceMesh::Property_map<vertex_descriptor, Point_3> location = sm.points();
  for (SurfaceMesh::Edge_iterator edgeIter = sm.edges_begin(); edgeIter != sm.edges_end(); ++edgeIter){
      vertex_descriptor s = sm.source(edgeIter->halfedge());
      vertex_descriptor t = sm.target(edgeIter->halfedge());
      const Point_3& a = location[s];
      const Point_3& b = location[t];
      length = CGAL::sqrt(CGAL::squared_distance(a, b));
      ++count;
      if (init){
          meanEdgeLen = minEdgeLen = maxEdgeLen = length;
          init = false;
      }
      else{
          if (length < minEdgeLen) minEdgeLen = length;
          if (length > maxEdgeLen) maxEdgeLen = length;
      }
      meanEdgeLen += length;
  }
  meanEdgeLen /= count;
  std::cout << "Edge Statistics: " << minEdgeLen << " " << maxEdgeLen << " " << meanEdgeLen << "\n";
  // return maxEdgeLen;
}

void OurMeshData::SimplifyMesh() {
    SMS::Count_ratio_stop_predicate<SurfaceMesh> stop(0.2);
}

void OurMeshData::CreateSubMeshFromMesh(const OurMeshData::OurMeshDataStructure& wholeMeshData, const int& planeIdx)
{
    // This function create a new mesh from wholeMesh. High level steps:
    // - maintain a mapping M from wholeMesh's vertexIdx to new vertexIdx
    // - iterate over wholeMesh's faces
    //    - if there is vertex which has not been added, add that vertex and update M

    float coordX, coordY, coordZ;
    std::vector<char> flagsMapToNewVertexIdx(wholeMeshData.Points.cols(), 0);
    std::vector<SurfaceMesh::Vertex_index> mappedNewVertexIdx(wholeMeshData.Points.cols());

    for (int k = 0; k < wholeMeshData.pointTriangles.size(); k++) {
        for (int f = 0; f < wholeMeshData.pointTriangles[k].size(); f++) {

            if (wholeMeshData.planeIdxAfterMP[wholeMeshData.triangleGlobalIdx[k][f]] == planeIdx) {
                for (int v = 0; v < wholeMeshData.pointTriangles[k][f].length; v++) {
                    if (flagsMapToNewVertexIdx[wholeMeshData.pointTriangles[k][f].c[v]] == 0) {
                        int oldVertexIdx = wholeMeshData.pointTriangles[k][f].c[v];
                        coordX = wholeMeshData.Points(0, oldVertexIdx);
                        coordY = wholeMeshData.Points(1, oldVertexIdx);
                        coordZ = wholeMeshData.Points(2, oldVertexIdx);
                        flagsMapToNewVertexIdx[oldVertexIdx] = 1;
                        mappedNewVertexIdx[oldVertexIdx] = sm.add_vertex(Point_3(coordX, coordY, coordZ));
                    }
                }
    
                // add face to new mesh after ensuring new mesh contains all vertices
                sm.add_face(mappedNewVertexIdx[wholeMeshData.pointTriangles[k][f].c[0]],
                            mappedNewVertexIdx[wholeMeshData.pointTriangles[k][f].c[1]],
                            mappedNewVertexIdx[wholeMeshData.pointTriangles[k][f].c[2]]);
            }
        }
    }
}

void OurMeshData::SubdivideMesh(const int NumberOfIterations, const int& globalDebugFlag, const std::string& prompt) {
    if (NumberOfIterations > 0) {
        CGAL::Subdivision_method_3::Loop_subdivision(sm, CGAL::parameters::number_of_iterations(NumberOfIterations));
        std::cout << prompt + "complete " << NumberOfIterations << " iters of sub-division" << std::endl;
    }
}

void OurMeshData::FlattenMesh(const int& globalDebugFlag, const std::string& prompt, int flagWriteUV)
{
    auto start = std::chrono::high_resolution_clock::now();

    int nCones = 100;
    bool flattenToDisk = false;
    bool mapToSphere = false;
    bool normalizeUVs = true;

    // load model
    bff::Model model;
    std::vector<bool> surfaceIsClosed;
    // loadModel(inputPath, model, surfaceIsClosed);
    loadModel(model, surfaceIsClosed, globalDebugFlag, prompt);

    // set nCones to 8 for closed surfaces
    for (int i = 0; i < model.size(); i++) {
        if (surfaceIsClosed[i] && !mapToSphere && nCones < 3) {
            std::cout << "Setting nCones to 8." << std::endl;
            nCones = 8;
        }
    }

    // flatten
    flatten(model, surfaceIsClosed, nCones, flattenToDisk, mapToSphere);

    data.resize(model.nVertices(), model.size());
    convertBFFMesh(model, surfaceIsClosed, mapToSphere, normalizeUVs);

    BFFPostProcessing(globalDebugFlag, prompt);

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << prompt + "complete flattening mesh in " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms" << std::endl;
}

void OurMeshData::BFFPostProcessing(const int& globalDebugFlag, const std::string& prompt)
{
    data.uvCnts.push_back(0);
    for (int k = 1; k < data.uvPts.size(); k++) {
        data.uvCnts.push_back(data.uvCnts[k - 1] + data.uvPts[k - 1].size());
    }

    int tmp_cnt = 0;
    for (int k = 0; k < data.textureTriangles.size(); k++) {
        std::vector<int> tmpTriangelIdx;
        for (int f = 0; f < data.textureTriangles[k].size(); f++) {
            tmpTriangelIdx.push_back(tmp_cnt);
            tmp_cnt++;
        }
        data.triangleGlobalIdx.push_back(tmpTriangelIdx);
    }

    for (int k = 0; k < data.textureTriangles.size(); k++) {
        assert (data.textureTriangles[k].size() == data.pointTriangles[k].size());
        data.nTextureTriangles += data.textureTriangles[k].size();
        data.nPointTriangles += data.pointTriangles[k].size();
    }

    if (globalDebugFlag == 1) {
        std::cout << prompt + "[Debug] #vertices: " << data.Points.cols() << std::endl;
        int debug_face_cnt = 0;
        for (auto it: data.pointTriangles) debug_face_cnt += it.size();
        std::cout << prompt + "[Debug] #faces: " << debug_face_cnt << std::endl;
        int debug_uv_cnt = 0;
        for (auto it: data.uvPts) debug_uv_cnt += it.size();
        std::cout << prompt + "[Debug] #uv: " << debug_uv_cnt << std::endl;

        // std::cout << prompt + "[Debug] uv cumulative cnts: ";
        // for (auto it: data.uvCnts) std::cout << it << " ";
        // std::cout << std::endl;

        double uv_sum = 0.0;
        double uv_min = std::numeric_limits<double>::min();
        double uv_max = std::numeric_limits<double>::max();
        std::cout << prompt + "[Debug] double numeric limit " << uv_min << " " << uv_max << std::endl;
        for (int i = 0; i < data.uvPts.size(); i++) {
            for (int j = 0; j < data.uvPts[i].size(); j++) {
                uv_sum += data.uvPts[i][j](0) + data.uvPts[i][j](1);
                if (uv_min > data.uvPts[i][j](0)) uv_min = data.uvPts[i][j](0);
                if (uv_max < data.uvPts[i][j](0)) uv_max = data.uvPts[i][j](0);
                if (uv_min > data.uvPts[i][j](1)) uv_min = data.uvPts[i][j](1);
                if (uv_max < data.uvPts[i][j](1)) uv_max = data.uvPts[i][j](1);
            }
        }
        std::cout << prompt + "[Debug] uv min_max " << uv_min << " " << uv_max << std::endl;
        std::cout << prompt + "[Debug] uv sum " << uv_sum << std::endl;
    }
}

void OurMeshData::ComputeTriangleAdjacency3DInBFF(bff::Model& model, const int& globalDebugFlag, const std::string& prompt) {
    // std::vector<std::vector<std::set<int>>> Adjacency(model.size(), std::vector<std::set<int> >());

    auto start = std::chrono::high_resolution_clock::now();

    data.infoMeshTriangleAdjacency3D.resize(model.size(), std::vector<std::set<unsigned int>>());

    int totalCnt = 0;
    // std::vector<int> maxNeighborFaceIdx(model.size(), -1);

    for(int k = 0; k < model.size(); ++k) {
        const bff::Mesh& mesh = model[k];
        int NumberOfFaces = mesh.faces.size();
        data.infoMeshTriangleAdjacency3D[k].resize(NumberOfFaces, std::set<unsigned int>());
        int faceCounter = 0;

        for (bff::FaceCIter f = mesh.faces.begin(); f != mesh.faces.end(); f++) {

            // NOTE, must check whether faces is used to fill hole.
            // Such face does not exist in original mesh from raw stream.
            if(!f->fillsHole) { //not sure about this? we don't write those to output, but they may be taken into account for flattening

                totalCnt++;

                int fIX = f->index;         // fIX is local index within each model
                assert(fIX == faceCounter);
                ++faceCounter;
                
                bff::HalfEdgeCIter h = f->halfEdge();
                do {
                    bff::HalfEdgeCIter nh = h->flip();

                    // NOTE must check whether the face is used to fill holes
                    if((!nh->onBoundary) && (!nh->face()->fillsHole)) {
                        int neighborIX = nh->face()->index;
                        data.infoMeshTriangleAdjacency3D[k][fIX].insert(neighborIX);
                        data.infoMeshTriangleAdjacency3D[k][neighborIX].insert(fIX);

                        // if (globalDebugFlag == 1) {
                        //     if (neighborIX > maxNeighborFaceIdx[k]) maxNeighborFaceIdx[k] = neighborIX;
                        // }
                    }
                    h = h->next();
                } while (h != f->halfEdge());
            }
        }
    }

    assert(data.nMeshTriangleAdjacency3D == 0);
    for (int k = 0; k < data.infoMeshTriangleAdjacency3D.size(); k++) {
        for (int f = 0; f < data.infoMeshTriangleAdjacency3D[k].size(); f++) {
            for (auto it: data.infoMeshTriangleAdjacency3D[k][f]) {
                data.nMeshTriangleAdjacency3D++;
            }
        }
    }
    data.nMeshTriangleAdjacency3D /= 2;
    std::cout << prompt + "3D mesh #adjcency: " << data.nMeshTriangleAdjacency3D << std::endl;

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << prompt + "complete computing adjacency matrix in " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms." << std::endl;

}

void OurMeshData::ComputeTriangleAdjacency2D(const int& globalDebugFlag, const std::string& prompt)
{
    auto start = std::chrono::high_resolution_clock::now();
    
    data.infoTextureTriangleAdjacency2D.resize(data.infoMeshTriangleAdjacency3D.size(), std::vector<std::set<unsigned int>>());

    int placeholder1, placeholder2;

    for (int k = 0; k < data.infoTextureTriangleAdjacency2D.size(); k++) {
        data.infoTextureTriangleAdjacency2D[k].resize(data.infoMeshTriangleAdjacency3D[k].size(), std::set<unsigned int>());

        for (int f = 0; f < data.infoMeshTriangleAdjacency3D[k].size(); f++) {
            for (auto it: data.infoMeshTriangleAdjacency3D[k][f]) {
                if (data.triangleGlobalIdx[k][f] < data.triangleGlobalIdx[k][it]) {
                    // we only add upper triangular information to avoid duplication.
                    // Since we conduct this process on each model component separately, we use local index instead of global one.
                    // if (CheckAdjacency2D(k, f, it, nEdgeCombine, edgeCombination)) {
                    if (NS_Utils::GetTwoCommonElementsInTriple(data.textureTriangles[k][f], data.textureTriangles[k][it], placeholder1, placeholder2)) {
                        data.infoTextureTriangleAdjacency2D[k][f].insert(it);
                        data.infoTextureTriangleAdjacency2D[k][it].insert(f);
                    }
                }
            }
        }
    }

    assert(data.nTextureTriangleAdjacency2D == 0);
    for (int k = 0; k < data.infoTextureTriangleAdjacency2D.size(); k++) {
        for (int f = 0; f < data.infoTextureTriangleAdjacency2D[k].size(); f++) {
            for (auto it: data.infoTextureTriangleAdjacency2D[k][f]) {
                data.nTextureTriangleAdjacency2D++;
            }
        }
    }
    data.nTextureTriangleAdjacency2D /= 2;
    std::cout << prompt + "2D texture #adjcency: " << data.nTextureTriangleAdjacency2D << std::endl;

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << prompt + "complete computing 2D texture adjacency in " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms" << std::endl;
}


void OurMeshData::ComputeTriangleAdjacency3D(const int& globalDebugFlag, const std::string& prompt)
{
    auto start = std::chrono::high_resolution_clock::now();

    // [#vertices, #vertices, X], X = #shared_faces * 2
    std::vector<std::vector<std::vector<int>>> faceEdgeShareInfos(data.Points.cols(), std::vector<std::vector<int>>());
    // iniitalize to -1
    // for (int i = 0; i < faceEdgeShareInfos.size(); i++) {
    //     std::cout << faceEdgeShareInfos[i].size() << std::endl;
    //     faceEdgeShareInfos[i].resize(data.Points.cols(), std::vector<int>());
    // }

    int nPtCombine = 3;
    int ptCombination[3][2] = {{0, 1}, {0, 2}, {1, 2}};

    for (int k = 0; k < data.pointTriangles.size(); k++) {
        for (int f = 0; f < data.pointTriangles[k].size(); f++) {
            for (int i = 0; i < nPtCombine; i++) {
                // we store edge information in upper triangular
                int ptIdx1, ptIdx2;
                if (data.pointTriangles[k][f].c[ptCombination[i][0]] > data.pointTriangles[k][f].c[ptCombination[i][1]]) {
                    ptIdx1 = data.pointTriangles[k][f].c[ptCombination[i][1]];
                    ptIdx2 = data.pointTriangles[k][f].c[ptCombination[i][0]];
                } else {
                    ptIdx1 = data.pointTriangles[k][f].c[ptCombination[i][0]];
                    ptIdx2 = data.pointTriangles[k][f].c[ptCombination[i][1]];
                }

                // std::cout << data.Points.cols() << " " << ptIdx1 << " " << ptIdx2 << " " << faceEdgeShareInfos.size() << std::endl;
                assert(ptIdx1 <= faceEdgeShareInfos.size());
                assert(ptIdx1 <= ptIdx2);

                if (faceEdgeShareInfos[ptIdx1].size() == 0) {
                    faceEdgeShareInfos[ptIdx1].resize(data.Points.cols(), std::vector<int>());
                }

                faceEdgeShareInfos[ptIdx1][ptIdx2].push_back(k);
                faceEdgeShareInfos[ptIdx1][ptIdx2].push_back(f);
            }
        }
    }
    
    data.infoMeshTriangleAdjacency3D.resize(data.pointTriangles.size(), std::vector<std::set<unsigned int>>());
    for (int k = 0; k < data.pointTriangles.size(); k++) {
        data.infoMeshTriangleAdjacency3D[k].resize(data.pointTriangles[k].size(), std::set<unsigned int>());
    }

    int tmpAdjCnt = 0;
    for (int ptIdx1 = 0; ptIdx1 < faceEdgeShareInfos.size(); ptIdx1++) {
        for (int ptIdx2 = ptIdx1; ptIdx2 < faceEdgeShareInfos[ptIdx1].size(); ptIdx2++) {
            // each elemenent stores (model_idx1, face1, model_idx2, face2)
            if (faceEdgeShareInfos[ptIdx1][ptIdx2].size() > 2) {
                // ensure faces come from same model
                assert(faceEdgeShareInfos[ptIdx1][ptIdx2][0] == faceEdgeShareInfos[ptIdx1][ptIdx2][2]);

                int model = faceEdgeShareInfos[ptIdx1][ptIdx2][0];
                int face1 = faceEdgeShareInfos[ptIdx1][ptIdx2][1];
                int face2 = faceEdgeShareInfos[ptIdx1][ptIdx2][3];
                data.infoMeshTriangleAdjacency3D[model][face1].insert(face2);
                data.infoMeshTriangleAdjacency3D[model][face2].insert(face1);

                tmpAdjCnt++;
            }
        }
    }

    assert (data.nMeshTriangleAdjacency3D == 0);
    for (int k = 0; k < data.infoMeshTriangleAdjacency3D.size(); k++) {
        for (int f = 0; f < data.infoMeshTriangleAdjacency3D[k].size(); f++) {
            for (auto it: data.infoMeshTriangleAdjacency3D[k][f]) data.nMeshTriangleAdjacency3D++;
        }
    }
    data.nMeshTriangleAdjacency3D /= 2;

    assert (tmpAdjCnt == data.nMeshTriangleAdjacency3D);

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << prompt << "complete finding 3D mesh #adj " << data.nMeshTriangleAdjacency3D \
              << " from #faces " << data.nPointTriangles << " in " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms." << std::endl;
}


void OurMeshData::loadModel(bff::Model& model, std::vector<bool>& surfaceIsClosed, const int& globalDebugFlag, const std::string& prompt) {
    std::string error;
    //if (bff::MeshIO::read(inputPath, model, error)) {
    if(ConvertCGAL2BFF(model, error)) {
        int nMeshes = model.size();
        surfaceIsClosed.resize(nMeshes, false);

        // NOTE, must compute adjacency before BFF fills holes or cuts the mesh
        ComputeTriangleAdjacency3DInBFF(model, globalDebugFlag, prompt);

        for (int i = 0; i < nMeshes; i++) {
            bff::Mesh& mesh = model[i];
            int nBoundaries = (int)mesh.boundaries.size();

            if (nBoundaries >= 1) {
                // mesh has boundaries
                int eulerPlusBoundaries = mesh.eulerCharacteristic() + nBoundaries;

                if (eulerPlusBoundaries == 2) {
                    // fill holes if mesh has more than 1 boundary
                    if (nBoundaries > 1) {
                        if (bff::HoleFiller::fill(mesh)) {
                            // all holes were filled
                            surfaceIsClosed[i] = true;
                        }
                    }

                } else {
                    // mesh probably has holes and handles
                    bff::HoleFiller::fill(mesh, true);
                    bff::Generators::compute(mesh);
                }

            } else if (nBoundaries == 0) {
                if (mesh.eulerCharacteristic() == 2) {
                    // mesh is closed
                    surfaceIsClosed[i] = true;

                } else {
                    // mesh has handles
                    bff::Generators::compute(mesh);
                }
            }
        }

    } else {
        std::cerr << "Unable to convert model. " << error << std::endl;
        exit(EXIT_FAILURE);
    }
}

bool OurMeshData::convertC2B(bff::Model& model, std::string& error) {
    bff::PolygonSoup soup;
    std::string line;
    int nVertices = 0;
    bool seenFace = false;
    std::set<std::pair<int, int>> uncuttableEdges;

    SurfaceMesh::Property_map<vertex_descriptor, Point_3> location = sm.points();
    for(vertex_descriptor vd : sm.vertices()) {
        double x = location[vd][0], y = location[vd][1], z = location[vd][2];
        soup.positions.emplace_back(bff::Vector(x, y, z));

        if (seenFace) {
            nVertices = 0;
            seenFace = false;
        }
        nVertices++;
    }
    for(auto iter=sm.faces_begin();iter!=sm.faces_end();++iter) {
        seenFace = true;
        int vertexCount = 0;
        int rootIndex = 0;
        int prevIndex = 0;
        CGAL::Vertex_around_face_iterator<SurfaceMesh> vbegin, vend;
        for(boost::tie(vbegin, vend) = vertices_around_face(sm.halfedge(*iter), sm);vbegin != vend;++vbegin) {
            int index = vbegin->idx();

            if (vertexCount == 0) rootIndex = index;

            vertexCount++;
            if (vertexCount > 3) {
                // triangulate polygon if vertexCount > 3
                soup.indices.emplace_back(rootIndex);
                soup.indices.emplace_back(prevIndex);
                soup.indices.emplace_back(index);

                // tag edge as uncuttable
                int i = rootIndex;
                int j = prevIndex;
                if (i > j) std::swap(i, j);
                std::pair<int, int> edge(i, j);
                uncuttableEdges.emplace(edge);

            } else {
                soup.indices.emplace_back(index);
            }

            prevIndex = index;
        }
    }

    // construct table
    soup.table.construct(soup.positions.size(), soup.indices);
    std::vector<int> isCuttableEdge(soup.table.getSize(), 1);
    for (std::set<std::pair<int, int>>::iterator it = uncuttableEdges.begin();
                                                 it != uncuttableEdges.end();
                                                 it++) {
        int eIndex = soup.table.getIndex(it->first, it->second);
        isCuttableEdge[eIndex] = 0;
    }

    // separate model into components
    std::vector<bff::PolygonSoup> soups;
    std::vector<std::vector<int>> isCuttableEdgeSoups;
    bff::MeshIO::separateComponents(soup, isCuttableEdge, soups, isCuttableEdgeSoups,
                       model.modelToMeshMap, model.meshToModelMap);

    // build halfedge meshes
    model.meshes.resize(soups.size());
    for (int i = 0; i < (int)soups.size(); i++) {
    //for(int i=0;i<1;i++) {
        int retVal = bff::MeshIO::buildMesh(soups[i], isCuttableEdgeSoups[i], model[i], error);
        if (retVal<0) {
            return false;
        } else if(retVal>0) {
            // adjust model maps as we increased the number of vertices
            // std::cout << "Adjusting vertices." << std::endl;
            for(int k=0;k<retVal;++k) {
                model.meshToModelMap[i].emplace_back(nVertices++);
                int index = (int)soups[i].positions.size()+k;
                model.modelToMeshMap.emplace_back(std::make_pair(i, index));
            }
        }
    }

    return true;
}


bool OurMeshData::ConvertCGAL2BFF(bff::Model& model, std::string& error) {

    bool success = false;
    if ((success = convertC2B(model, error))) {
        bff::MeshIO::normalize(model);
    }
    return success;
}


void OurMeshData::flatten(bff::Model& model, const std::vector<bool>& surfaceIsClosed,
             int nCones, bool flattenToDisk, bool mapToSphere) {

    int nMeshes = model.size();

    for (int i = 0; i < nMeshes; i++) {
        bff::Mesh& mesh = model[i];
        bff::BFF bff(mesh);

        if (nCones > 0) {
            std::vector<bff::VertexIter> cones;
            bff::DenseMatrix coneAngles(bff.data->iN);
            int S = std::min(nCones, (int)mesh.vertices.size() - bff.data->bN);

            if (bff::ConePlacement::findConesAndPrescribeAngles(S, cones, coneAngles, bff.data, mesh)
                == bff::ConePlacement::ErrorCode::ok) {
                if (!surfaceIsClosed[i] || cones.size() > 0) {
                    bff::Cutter::cut(cones, mesh);
                    bff.flattenWithCones(coneAngles, true);
                }
            }

        } else {
            if (surfaceIsClosed[i]) {
                if (mapToSphere) {
                    bff.mapToSphere();

                } else {
                    std::cerr << "Surface is closed. Either specify nCones or mapToSphere." << std::endl;
                    exit(EXIT_FAILURE);
                }

            } else {
                if (flattenToDisk) {
                    bff.flattenToDisk();

                } else {
                    bff::DenseMatrix u(bff.data->bN);
                    bff.flatten(u, true);
                }
            }
        }
    }
}

void OurMeshData::convertBFFMesh(bff::Model& model, const std::vector<bool>& surfaceIsClosed, bool mapToSphere, bool normalize) {
    int nMeshes = model.size();
    std::vector<bool> mappedToSphere(nMeshes, false);
    for (int i = 0; i < nMeshes; i++) {
        if (surfaceIsClosed[i]) {
            mappedToSphere[i] = mapToSphere;
        }
    }

    // pack
    std::vector<bff::Vector> originalCenters, newCenters;
    std::vector<bool> flippedBins;
    bff::Vector modelMinBounds, modelMaxBounds;
    bff::BinPacking::pack(model, mappedToSphere, originalCenters, newCenters,
                     flippedBins, modelMinBounds, modelMaxBounds);

    // write vertex positions
    for (int i = 0; i < model.nVertices(); i++) {
        std::pair<int, int> vData = model.localVertexIndex(i);
        const bff::Mesh& mesh = model[vData.first];
        bff::VertexCIter v = mesh.vertices.begin() + vData.second;

        bff::Vector p = v->position*mesh.radius + mesh.cm;
        data.Points(0, i) = p.x;
        data.Points(1, i) = p.y;
        data.Points(2, i) = p.z;
    }

    // write uvs and indices
    int nUvs = 0;
    for (int i = 0; i < model.size(); i++) {
        // compute uv radius and shift
        bff::Vector minExtent(modelMinBounds.x, modelMinBounds.y);
        double dx = modelMaxBounds.x - minExtent.x;
        double dy = modelMaxBounds.y - minExtent.y;
        double extent = std::max(dx, dy);
        minExtent.x -= (extent - dx)/2.0;
        minExtent.y -= (extent - dy)/2.0;

        // compute sphere radius if component has been mapped to a sphere
        double sphereRadius = 1.0;
        if (mappedToSphere[i]) {
            for (bff::WedgeCIter w = model[i].wedges().begin(); w != model[i].wedges().end(); w++) {
                sphereRadius = std::max(w->uv.norm(), sphereRadius);
            }
        }

        // write vertices and interior uvs
        int uvCount = 0;
        bff::HalfEdgeData<int> uvIndexMap(model[i]);

        for (bff::VertexCIter v = model[i].vertices.begin(); v != model[i].vertices.end(); v++) {
            if (!v->onBoundary()) {
                Eigen::Vector2d uvPt = getUV(v->wedge()->uv, mappedToSphere[i], sphereRadius,
                        model[i].radius, originalCenters[i], newCenters[i],
                        minExtent, extent, flippedBins[i], normalize);
                data.uvPts[i].push_back(uvPt);

                bff::HalfEdgeCIter he = v->halfEdge();
                do {
                    uvIndexMap[he->next()] = uvCount;

                    he = he->flip()->next();
                } while (he != v->halfEdge());

                uvCount++;
            }
        }

        // write boundary uvs
        for (bff::WedgeCIter w: model[i].cutBoundary()) {
            Eigen::Vector2d uvPt = getUV(w->uv, mappedToSphere[i], sphereRadius,
                    model[i].radius, originalCenters[i], newCenters[i],
                    minExtent, extent, flippedBins[i], normalize);
            data.uvPts[i].push_back(uvPt);

            bff::HalfEdgeCIter he = w->halfEdge()->prev();
            do {
                uvIndexMap[he->next()] = uvCount;

                if (he->edge()->onCut) break;
                he = he->flip()->next();
            } while (!he->onBoundary);

            uvCount++;
        }

        // write indices
        int uncuttableEdges = 0;
        for (bff::FaceCIter f = model[i].faces.begin(); f != model[i].faces.end(); f++) {
            if (!f->fillsHole) {
                if (uncuttableEdges > 0) {
                    uncuttableEdges--;
                    continue;
                }

                //writeString(out, "f");

                bff::HalfEdgeCIter he = f->halfEdge()->next();
                while (!he->edge()->isCuttable) he = he->next();
                bff::HalfEdgeCIter fhe = he;
                std::unordered_map<int, bool> seenUncuttableEdges;

                NS_Utils::Triple pT, tT;
                int cnt = 0;

                do {
                    if(cnt==3) {
                        std::cout << "Not a triangle mesh" << std::endl;
                        exit(0);
                    }
                    bff::VertexCIter v = he->vertex();
                    int vIndex = v->referenceIndex == -1 ? v->index : v->referenceIndex;
                    //writeString(out, " " + std::to_string(model.globalVertexIndex(i, vIndex) + 1) + "/" +
                    //					   std::to_string(nUvs + uvIndexMap[he->next()] + 1));
                    pT.c[cnt] = model.globalVertexIndex(i, vIndex);
                    tT.c[cnt++] = nUvs + uvIndexMap[he->next()];

                    he = he->next();
                    while (!he->edge()->isCuttable) {
                        seenUncuttableEdges[he->edge()->index] = true;
                        he = he->flip()->next();
                    }

                } while (he != fhe);

                data.pointTriangles[i].push_back(pT);
                data.textureTriangles[i].push_back(tT);

                uncuttableEdges = (int)seenUncuttableEdges.size();

                //writeString(out, "\n");
            }
        }

        nUvs += uvCount;
    }
}

void OurMeshData::writeModelUVs(const std::string& outputPath, bff::Model& model,
                   const std::vector<bool>& surfaceIsClosed, bool mapToSphere,
                   bool normalizeUVs)
{
    int nMeshes = model.size();
    std::vector<bool> mappedToSphere(nMeshes, false);
    for (int i = 0; i < nMeshes; i++) {
        if (surfaceIsClosed[i]) {
            mappedToSphere[i] = mapToSphere;
        }
    }

    if (!bff::MeshIO::write(outputPath, model, mappedToSphere, normalizeUVs)) {
        std::cerr << "Unable to write file: " << outputPath << std::endl;
        exit(EXIT_FAILURE);
    }
}

Eigen::Vector2d OurMeshData::getUV(bff::Vector uv, bool mapToSphere, double sphereRadius,
             double meshRadius, const bff::Vector& oldCenter, const bff::Vector& newCenter,
             const bff::Vector& minExtent, double extent, bool flipped, bool normalize)
{
    // resize
    if (mapToSphere) {
        uv /= sphereRadius;
        uv.x = 0.5 + atan2(uv.z, uv.x)/(2*M_PI);
        uv.y = 0.5 - asin(uv.y)/M_PI;
    } else {
        uv *= meshRadius;
    }

    // shift
    uv -= oldCenter;
    if (flipped) uv = bff::Vector(-uv.y, uv.x);
    uv += newCenter;
    uv -= minExtent;
    if (normalize) uv /= extent;

    return Eigen::Vector2d(uv.x, uv.y);
}

void OurMeshData::TextureTriOverlapCheck(std::string& prompt) {

    std::cout << prompt + "Start checking triangle overlapping (" << data.nTextureTriangles << ")..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    // each row stores (base.u, base.v, dir1.u, dir1.v, dir2.u, dir2.v, invdet)
    Eigen::Array<float, Eigen::Dynamic, 7> ptInTriPreCompute(data.nTextureTriangles, 7);

    {
    float base_u, base_v, dir1_u, dir1_v, dir2_u, dir2_v, invdet;

    for (int k = 0; k < data.textureTriangles.size(); k++) {
        for (int f = 0; f < data.textureTriangles[k].size(); f++) {
            base_u = (float)data.uvPts[k][data.textureTriangles[k][f].c[0] - data.uvCnts[k]](0);
            base_v = (float)data.uvPts[k][data.textureTriangles[k][f].c[0] - data.uvCnts[k]](1);
            dir1_u = (float)data.uvPts[k][data.textureTriangles[k][f].c[1] - data.uvCnts[k]](0) - base_u;
            dir1_v = (float)data.uvPts[k][data.textureTriangles[k][f].c[1] - data.uvCnts[k]](1) - base_v;
            dir2_u = (float)data.uvPts[k][data.textureTriangles[k][f].c[2] - data.uvCnts[k]](0) - base_u;
            dir2_v = (float)data.uvPts[k][data.textureTriangles[k][f].c[2] - data.uvCnts[k]](1) - base_v;
            invdet = 1.f / (dir1_u * dir2_v - dir1_v * dir2_u);
            
            Eigen::Array<float, 1, 7> tmp;
            tmp << base_u, base_v, dir1_u, dir1_v, dir2_u, dir2_v, invdet;
            ptInTriPreCompute(data.triangleGlobalIdx[k][f], Eigen::all) = tmp;
        }
    }
    }

    // each element (f1_p1, f1_p2, f2_p1, f2_p2) represents two edges, which come from two faces
    int nEdgeCombine = 9;
    int edgeCombination[9][4] = {{0, 1, 0, 1}, {0, 1, 0, 2}, {0, 1, 1, 2},
                                 {0, 2, 0, 1}, {0, 2, 0, 2}, {0, 2, 1, 2},
                                 {1, 2, 0, 1}, {1, 2, 0, 2}, {1, 2, 1, 2}};

    // Assume we have a flag matrix with size (#TextureTriangles, #TextureTriangles),
    // we compare all texture triangle pair in matrix's lower triangular area in F-order.
    // Namely, we compare triangel 2 ~ N to 1, then compare triangel 3 ~ N to 2, ...
    // If one triangle A has been found overlapped with another one B,
    // we will not compare triangles later to A since A's texture will not appear in the final material image.
    // Essentially, when overlapping happens, we put aside triangles with higher index.

    // DEBUG
    // std::vector<int> debugTriOverlapIdx(data.flagsTextureTriangleOverlap.size(), -1);

    data.infoTextureTriangleOverlap.resize(data.textureTriangles.size(), std::vector<std::vector<unsigned int>>());
    for (int k = 0; k < data.textureTriangles.size(); k++) {
        data.infoTextureTriangleOverlap[k].resize(data.textureTriangles[k].size(), std::vector<unsigned int>());
    }

    assert (data.nTextureTriangleOverlaps == 0);
    assert (data.maxTimesTextureTriangleOverlap == 0);
    data.flagsTextureTriangleOverlap.resize(data.nTextureTriangles, 0);

    // create a map from global index to a pair of (modelIdx, faceIdx)
    std::vector<std::pair<unsigned int, unsigned int>> triangleGlobalIdxToPairIdx(data.nTextureTriangles, std::pair<unsigned int, unsigned int>());
    for (int k = 0; k < data.textureTriangles.size(); k++) {
        for (int f = 0; f < data.textureTriangles[k].size(); f++) {
            triangleGlobalIdxToPairIdx[data.triangleGlobalIdx[k][f]] = std::make_pair(k, f);
        }
    }

    std::vector<Eigen::Triplet<bool>> tripletList;

#pragma omp parallel 
{
    std::vector<Eigen::Triplet<bool>> tripletListThread;
#pragma omp for schedule(dynamic, 10)
    for (int global_f1 = 0; global_f1 < data.nTextureTriangles; global_f1++) {

        if (global_f1 % 500==0) {
            std::cout << "[overlap check] thread " << omp_get_thread_num() << " " << global_f1 << "/" << data.nTextureTriangles << std::endl;
        }

        float base_u, base_v, dir1_u, dir1_v, dir2_u, dir2_v, invdet;

        for (int global_f2 = global_f1 + 1; global_f2 < data.nTextureTriangles; global_f2++) {

            int k1 = triangleGlobalIdxToPairIdx[global_f1].first;
            int f1 = triangleGlobalIdxToPairIdx[global_f1].second;
            assert (global_f1 == data.triangleGlobalIdx[k1][f1]);

            int k2 = triangleGlobalIdxToPairIdx[global_f2].first;
            int f2 = triangleGlobalIdxToPairIdx[global_f2].second;
            assert (global_f2 == data.triangleGlobalIdx[k2][f2]);

            // Only check the upper triangular area
            //if (global_f1 < global_f2) {

                // this flag is used to indicate whether (k1, f1) and (k2, f2) overlap
                int flagOverlap = 0;
                
                // Check 1: whether face2's vertices are in face1
                if (flagOverlap == 0) {
                    base_u = ptInTriPreCompute(global_f1, 0);
                    base_v = ptInTriPreCompute(global_f1, 1);
                    dir1_u = ptInTriPreCompute(global_f1, 2);
                    dir1_v = ptInTriPreCompute(global_f1, 3);
                    dir2_u = ptInTriPreCompute(global_f1, 4);
                    dir2_v = ptInTriPreCompute(global_f1, 5);
                    invdet = ptInTriPreCompute(global_f1, 6);
                    for (int i = 0; i < data.textureTriangles[k2][f2].length; i++) {
                        float pt_dir_u = (float)data.uvPts[k2][data.textureTriangles[k2][f2].c[i] - data.uvCnts[k2]](0) - base_u;
                        float pt_dir_v = (float)data.uvPts[k2][data.textureTriangles[k2][f2].c[i] - data.uvCnts[k2]](1) - base_v;
                        float n1 = (dir2_v * pt_dir_u - dir2_u * pt_dir_v) * invdet;
                        float n2 = (-dir1_v * pt_dir_u + dir1_u * pt_dir_v) * invdet;
                        if (n1 > 0 && n2 > 0 && n1 + n2 < 1) {
                            // Note: be careful, we need to exclude overlapping on vertex, namely n1 = 0 or n2 = 0
                            flagOverlap = 1;
                            break;
                        }
                    }
                }
                
                // Check 2: whether face1's vertices are in face2
                if (flagOverlap == 0) {
                    base_u = ptInTriPreCompute(global_f2, 0);
                    base_v = ptInTriPreCompute(global_f2, 1);
                    dir1_u = ptInTriPreCompute(global_f2, 2);
                    dir1_v = ptInTriPreCompute(global_f2, 3);
                    dir2_u = ptInTriPreCompute(global_f2, 4);
                    dir2_v = ptInTriPreCompute(global_f2, 5);
                    invdet = ptInTriPreCompute(global_f2, 6);
                    for (int i = 0; i < data.textureTriangles[k1][f1].length; i++) {
                        float pt_dir_u = (float)data.uvPts[k1][data.textureTriangles[k1][f1].c[i] - data.uvCnts[k1]](0) - base_u;
                        float pt_dir_v = (float)data.uvPts[k1][data.textureTriangles[k1][f1].c[i] - data.uvCnts[k1]](1) - base_v;
                        float n1 = (dir2_v * pt_dir_u - dir2_u * pt_dir_v) * invdet;
                        float n2 = (-dir1_v * pt_dir_u + dir1_u * pt_dir_v) * invdet;
                        if (n1 > 0 && n2 > 0 && n1 + n2 < 1) {
                            // Note: be careful, we need to exclude overlapping on vertex, namely n1 = 0 or n2 = 0
                            flagOverlap = 1;
                            break;
                        }
                    }
                }
                
                // Check 3: whether face1 and face2 have edge intersections
                if (flagOverlap == 0) {
                    for (int i = 0; i < nEdgeCombine; i++) {
                        // edge from face 1
                        float u1 = (float)data.uvPts[k1][data.textureTriangles[k1][f1].c[edgeCombination[i][0]] - data.uvCnts[k1]](0);
                        float v1 = (float)data.uvPts[k1][data.textureTriangles[k1][f1].c[edgeCombination[i][0]] - data.uvCnts[k1]](1);
                        float u2 = (float)data.uvPts[k1][data.textureTriangles[k1][f1].c[edgeCombination[i][1]] - data.uvCnts[k1]](0);
                        float v2 = (float)data.uvPts[k1][data.textureTriangles[k1][f1].c[edgeCombination[i][1]] - data.uvCnts[k1]](1);
                      
                        // edge from face 2
                        float u3 = (float)data.uvPts[k2][data.textureTriangles[k2][f2].c[edgeCombination[i][2]] - data.uvCnts[k2]](0);
                        float v3 = (float)data.uvPts[k2][data.textureTriangles[k2][f2].c[edgeCombination[i][2]] - data.uvCnts[k2]](1);
                        float u4 = (float)data.uvPts[k2][data.textureTriangles[k2][f2].c[edgeCombination[i][3]] - data.uvCnts[k2]](0);
                        float v4 = (float)data.uvPts[k2][data.textureTriangles[k2][f2].c[edgeCombination[i][3]] - data.uvCnts[k2]](1);
                           
                        if (NS_Utils::CheckLineSegIntersect2D(u1, v1, u2, v2, u3, v3, u4, v4)) {
                            flagOverlap = 1;
                            break;
                        }
                    }
                }
                
                if (flagOverlap == 1) {
                    // std::cout << "thread " << omp_get_thread_num() << " " << global_f1 << " " << global_f2 << std::endl;
                    //flagTextureTrianglePairOverlap(global_f1, global_f2) = true;
                    //flagTextureTrianglePairOverlap(global_f2, global_f1) = true;
                    tripletListThread.push_back(Eigen::Triplet<bool>(global_f1,global_f2,true));
                }
            //}
        }
    }
#pragma omp critical
{
    tripletList.insert(tripletList.end(), tripletListThread.begin(), tripletListThread.end());
}
}

    Eigen::SparseMatrix<bool> flagTextureTrianglePairOverlap(data.nTextureTriangles, data.nTextureTriangles);
    flagTextureTrianglePairOverlap.setFromTriplets(tripletList.begin(), tripletList.end());

    for (int global_f1 = 0; global_f1 < data.nTextureTriangles; global_f1++) {
        int k1 = triangleGlobalIdxToPairIdx[global_f1].first;
        int f1 = triangleGlobalIdxToPairIdx[global_f1].second;

        for (int global_f2 = global_f1 + 1; global_f2 < data.nTextureTriangles; global_f2++) {
            int k2 = triangleGlobalIdxToPairIdx[global_f2].first;
            int f2 = triangleGlobalIdxToPairIdx[global_f2].second;

            // Only check the upper triangular area
            if (flagTextureTrianglePairOverlap.coeff(global_f1, global_f2)) {
                data.nTextureTriangleOverlaps++;

                data.infoTextureTriangleOverlap[k1][f1].push_back(k2);
                data.infoTextureTriangleOverlap[k1][f1].push_back(f2);
                data.infoTextureTriangleOverlap[k2][f2].push_back(k1);
                data.infoTextureTriangleOverlap[k2][f2].push_back(f1);

                if (data.flagsTextureTriangleOverlap[global_f2] < 1 + data.flagsTextureTriangleOverlap[global_f1]) {
                    data.flagsTextureTriangleOverlap[global_f2] = 1 + data.flagsTextureTriangleOverlap[global_f1];

                    if (data.maxTimesTextureTriangleOverlap < data.flagsTextureTriangleOverlap[global_f2]) {
                        // NOTE: this number of maximum planes is not ideal.
                        // If two pairs, (f1, f2) and (f2, f3), overlap. We just simply make the maximum number as 3.
                        // However, f1 and f3 may not overlap. Hope this approximation works well.
                        data.maxTimesTextureTriangleOverlap = data.flagsTextureTriangleOverlap[global_f2];
                    }
                }
            }
        }
    }

    assert(data.nTextureTrianglesWithOverlap == 0);
    for (auto it: data.flagsTextureTriangleOverlap) data.nTextureTrianglesWithOverlap += (it != 0);

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << prompt + "... find " << data.nTextureTriangleOverlaps \
              << " overlaps from " << data.nTextureTrianglesWithOverlap << "/" <<  data.nTextureTriangles \
              << " triangles (max " << data.maxTimesTextureTriangleOverlap << ")" \
              << " in " << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms" << std::endl;
}


void OurMeshData::CreatePlaceholderInfo()
{
    int nPlanes = data.maxTimesTextureTriangleOverlap + 1;
    assert(data.maxPlaneIdx == 0);
    data.nTextureTriangles = data.nPointTriangles;
    data.planeIdxAfterRePack.resize(data.nTextureTriangles, 0);
    data.nTextureTrianglesOnPlane.resize(nPlanes, 0);
}


void OurMeshData::ResolveOverlappedConformalMap(NS_MPSolver::MPSolver<float, int>& MP, const int& nItersMP,
                                                // const bool& flagPlaceHolder,
                                                const int& globalDebugFlag, const std::string& prompt)
{
    // maxTimesTextureTriangleOverlap is 0-based maximum index
    int nPlanes = data.maxTimesTextureTriangleOverlap + 1;
    
    if ((data.maxTimesTextureTriangleOverlap == 0)) {
        assert(data.maxPlaneIdx == 0);
        data.planeIdxAfterMP.resize(data.nTextureTriangles, 0);
        data.nTextureTrianglesOnPlane.resize(nPlanes, 0);
        return;
    }

    std::cout << prompt + "Start resolving TexTriangle overlapping with MP ... " << std::endl; 
    auto resolveOverlapStart = std::chrono::high_resolution_clock::now();

    MP.CreateGraph(data.nTextureTriangles, nPlanes);

    // assign 0.5 to the plane without conflict and 0.0 otherwise. We may try random values later.
    std::vector<float> infoUnary(data.nTextureTriangles * nPlanes, 0.0);
    for (int k = 0; k < data.textureTriangles.size(); k++) {
        for (int f = 0; f < data.textureTriangles[k].size(); f++) {
            int tmpPlaneID = data.flagsTextureTriangleOverlap[data.triangleGlobalIdx[k][f]];
            infoUnary[data.triangleGlobalIdx[k][f] * nPlanes + tmpPlaneID] = 0.5;
        }
    }
    std::vector<float> tmpInfoPair;
    std::vector<float> tmpPotentialsPair;

    MP.UpdateGraph(MP.GraphUpdateTypes::OnlyUnary, infoUnary, tmpInfoPair, tmpPotentialsPair);

    // Update faces's adjacency information
    std::vector<float> tmpInfoUnary;
    std::vector<float> potentialPairAdj(nPlanes * nPlanes, 0.0);
    for (int i = 0; i < nPlanes; i++) {
        // diagonal
        potentialPairAdj[i * nPlanes + i] = 10.0;
    }
    std::vector<float> infoPairAdj;
    int tmpAdjCnt = 0;
    for (int k = 0; k < data.infoMeshTriangleAdjacency3D.size(); k++) {
        for (int f = 0; f < data.infoMeshTriangleAdjacency3D[k].size(); f++) {
            for (auto it: data.infoMeshTriangleAdjacency3D[k][f]) {
                if (data.triangleGlobalIdx[k][f] < data.triangleGlobalIdx[k][it]) {
                    // we only add upper triangular information to avoid duplication
                    infoPairAdj.push_back(data.triangleGlobalIdx[k][f]);
                    infoPairAdj.push_back(data.triangleGlobalIdx[k][it]);
                    infoPairAdj.push_back(1.0);
                    tmpAdjCnt++;
                }
            }
        }
    }
    if (globalDebugFlag == 1) {
        std::cout << prompt + "[Debug] #meshAdj: " << data.nMeshTriangleAdjacency3D << "; #graphAdjInfo: " << tmpAdjCnt << std::endl;
    }
    assert(tmpAdjCnt == data.nMeshTriangleAdjacency3D);

    MP.UpdateGraph(MP.GraphUpdateTypes::OnlyPair, tmpInfoUnary, infoPairAdj, potentialPairAdj);

    // Update faces's overlapping information
    std::vector<float> potentialPairOverlap(nPlanes * nPlanes, 0.0);
    for (int row = 0; row < nPlanes; row++) {
        for (int col = 0; col < nPlanes; col++) {
            // off-diagonal
            if (row != col) potentialPairOverlap[row * nPlanes + col] = 10.0;
        }
    }
    std::vector<float> infoPairOverlap;
    int tmpOverlapCnt = 0;
    for (int k1 = 0; k1 < data.infoTextureTriangleOverlap.size(); k1++) {
        for (int f1 = 0; f1 < data.infoTextureTriangleOverlap[k1].size(); f1++) {
            assert (data.infoTextureTriangleOverlap[k1][f1].size() % 2 == 0);
            for (int i = 0; i < data.infoTextureTriangleOverlap[k1][f1].size(); i += 2) {
                int k2 = data.infoTextureTriangleOverlap[k1][f1][i];
                int f2 = data.infoTextureTriangleOverlap[k1][f1][i + 1];
                if (data.triangleGlobalIdx[k1][f1] < data.triangleGlobalIdx[k2][f2]) {
                    // we only add upper triangular information to avoid duplication
                    infoPairOverlap.push_back(data.triangleGlobalIdx[k1][f1]);
                    infoPairOverlap.push_back(data.triangleGlobalIdx[k2][f2]);
                    infoPairOverlap.push_back(1.0);
                    tmpOverlapCnt++;
                }
            }
        }
    }
    if (globalDebugFlag == 1) {
        std::cout << prompt + "[Debug] #textureOverlap: " << data.nTextureTriangleOverlaps << "; #graphOverlapInfo: " << tmpOverlapCnt << std::endl;
    }
    assert(tmpOverlapCnt == data.nTextureTriangleOverlaps);

    MP.UpdateGraph(MP.GraphUpdateTypes::OnlyPair, tmpInfoUnary, infoPairOverlap, potentialPairOverlap);

    // check all regions/connections are added
    int nExpextedRegions = data.nTextureTriangles + tmpAdjCnt + tmpOverlapCnt;
    int nExpextedLamdaSize = (tmpAdjCnt + tmpOverlapCnt) * nPlanes * 2;
    assert(MP.CheckGraph(nExpextedRegions, nExpextedLamdaSize, globalDebugFlag, prompt));

    MP.Solve(nItersMP);

    assert(data.maxPlaneIdx == 0);
    data.planeIdxAfterMP.resize(data.nTextureTriangles, 0);
    data.nTextureTrianglesOnPlane.resize(nPlanes, 0);
    for (int i = 0; i < MP.varState.size(); i++) {
        data.planeIdxAfterMP[i] = MP.varState[i];
        data.nTextureTrianglesOnPlane[MP.varState[i]] += 1;
        if (MP.varState[i] > data.maxPlaneIdx) data.maxPlaneIdx = MP.varState[i];
    }
    std::cout << prompt + "max planeID after MP: " << data.maxPlaneIdx << std::endl;

    // mannually assign still-overlapped triangles to different planes.
    for (int k1 = 0; k1 < data.infoTextureTriangleOverlap.size(); k1++) {
        for (int f1 = 0; f1 < data.infoTextureTriangleOverlap[k1].size(); f1++) {
            assert (data.infoTextureTriangleOverlap[k1][f1].size() % 2 == 0);
            for (int i = 0; i < data.infoTextureTriangleOverlap[k1][f1].size(); i += 2) {
                int k2 = data.infoTextureTriangleOverlap[k1][f1][i];
                int f2 = data.infoTextureTriangleOverlap[k1][f1][i + 1];
                
                if (data.planeIdxAfterMP[data.triangleGlobalIdx[k1][f1]] == data.planeIdxAfterMP[data.triangleGlobalIdx[k2][f2]]) {
                    // NOTE: because we have bin-packing later, to resolve overlap,
                    // we only need to assign still-on-same-plane overlapped traingles to different plaens.
                    std::vector<bool> availablePlanes(data.maxPlaneIdx + 1, true);
                    availablePlanes[data.planeIdxAfterMP[data.triangleGlobalIdx[k1][f1]]] = false;

                    for (int j = 0; j < data.infoTextureTriangleOverlap[k2][f2].size(); j += 2) {
                        int k3 = data.infoTextureTriangleOverlap[k2][f2][j];
                        int f3 = data.infoTextureTriangleOverlap[k2][f2][j + 1];
                        availablePlanes[data.planeIdxAfterMP[data.triangleGlobalIdx[k3][f3]]] = false;
                    }

                    int newPlaneIdx = -1;
                    for (int j = 0; j <= data.maxPlaneIdx; j++) {
                        if (availablePlanes[j])  {
                            newPlaneIdx = j;
                            break;
                        }
                    }
                    if (newPlaneIdx == -1) {
                        newPlaneIdx = data.maxPlaneIdx + 1;
                        data.maxPlaneIdx += 1;
                        data.maxTimesTextureTriangleOverlap += 1;
                        nPlanes += 1;
                    }
                    assert (newPlaneIdx < nPlanes);
                    data.nTextureTrianglesOnPlane[data.planeIdxAfterMP[data.triangleGlobalIdx[k1][f1]]] -= 1;
                    data.planeIdxAfterMP[data.triangleGlobalIdx[k2][f2]] = newPlaneIdx;
                }
            }
        }
    }

    std::cout << prompt + "max planeID after MP and manual assignment: " << data.maxPlaneIdx << std::endl;

    // double check it is overlap free
    for (int k1 = 0; k1 < data.infoTextureTriangleOverlap.size(); k1++) {
        for (int f1 = 0; f1 < data.infoTextureTriangleOverlap[k1].size(); f1++) {
            assert (data.infoTextureTriangleOverlap[k1][f1].size() % 2 == 0);
            for (int i = 0; i < data.infoTextureTriangleOverlap[k1][f1].size(); i += 2) {
                int k2 = data.infoTextureTriangleOverlap[k1][f1][i];
                int f2 = data.infoTextureTriangleOverlap[k1][f1][i + 1];
                assert (data.planeIdxAfterMP[data.triangleGlobalIdx[k1][f1]] != data.planeIdxAfterMP[data.triangleGlobalIdx[k2][f2]]);
            }
        }
    }

    int tmpDiff = 0;
    if (globalDebugFlag == 1) {
        for (int k = 0; k < data.textureTriangles.size(); k++) {
            for (int f = 0; f < data.textureTriangles[k].size(); f++) {
                if (MP.varState[data.triangleGlobalIdx[k][f]] != data.flagsTextureTriangleOverlap[data.triangleGlobalIdx[k][f]]) {
                    tmpDiff++;
                }
            }
        }
        std::cout << prompt + "[Debug] difference between heuristic plane assignment and MP results: " << tmpDiff << std::endl;

        std::ofstream outf1("/Users/apple/Desktop/Coding/C++/3DTexture/20200920/debug_resolve_overlap_heuristic.txt", std::ios_base::out);
        for (int i = 0; i < data.flagsTextureTriangleOverlap.size(); i++) {outf1 << i << " " << (int)data.flagsTextureTriangleOverlap[i] << std::endl;}
        outf1.close();

        std::ofstream outf2("/Users/apple/Desktop/Coding/C++/3DTexture/20200920/debug_resolve_overlap_mp.txt", std::ios_base::out);
        for (int i = 0; i < MP.varState.size(); i++) {outf2 << i << " " << MP.varState[i] << std::endl;}
        outf2.close();
    }

    auto resolveOverlapFinish = std::chrono::high_resolution_clock::now();
    std::cout << prompt + "Resolve TexTriangle overlapping with MP in " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(resolveOverlapFinish - resolveOverlapStart).count() << " ms" << std::endl;

    // do not forget to destroy graph
    MP.DestroyGraph();
}


void OurMeshData::BinPackTextureTriangles2D(const BinPackTypeEnum& binPackType, const bool& flagConformalMap, const int& nBinPackAreasPerPlate,
                                            const NS_StreamFile::StreamFile& streamInfos,
                                            const Eigen::MatrixXf& NDC, const std::vector<Eigen::MatrixXf>& faceAreaMatrix,
                                            const int& mtlHeight, const int& mtlWidth, const float& adapatThreshold,
                                            const int& globalDebugFlag, const std::string& prompt)
{
    std::cout << prompt + "Start bin-packing 2D texture triangles for compact material ..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<double> areas;
    std::vector<bff::Vector> minBounds;
    std::vector<bff::Vector> maxBounds;

    std::vector<std::vector<int>> visitedUVPts;
    std::vector<unsigned int> addUVCnts;

    if (flagConformalMap) {
        BinPackPreprocessForConformalMap(areas, minBounds, maxBounds, globalDebugFlag, prompt);

        addUVCnts.resize(data.uvCnts.size(), 0);
        data.uvPtsAfterRePack.resize(data.uvPts.size());
        visitedUVPts.resize(data.uvPts.size());
        for (int k = 0; k < data.uvPts.size(); k++) {
            data.uvPtsAfterRePack[k].resize(data.uvPts[k].size());
            visitedUVPts[k].resize(data.uvPts[k].size(), -1);
        }

        data.planeIdxAfterRePack.resize(data.planeIdxAfterMP.size(), 0);
    } else {
        bool flagAreaMesh = (binPackType == BinPackTypeEnum::FACE_AREA_ADAPTIVE) ? false : true;
        BinPackPreprocessForNonConformalMap(areas, minBounds, maxBounds, flagAreaMesh, streamInfos, NDC, faceAreaMatrix, globalDebugFlag, prompt);

        data.planeIdxAfterRePack.resize(data.nTextureTriangles, 0);
    }

    int nConnectedAreas = (int)areas.size();
    std::vector<std::vector<int>> vecAreaIdxs;

    if (binPackType == BinPackTypeEnum::FIX_NUM)
    {
        data.maxPlaneIdx = (int)std::ceil((float)nConnectedAreas / nBinPackAreasPerPlate) - 1;
    
        for (int planeIdx = 0; planeIdx <= data.maxPlaneIdx; planeIdx++) {
            int startAreaIdx = planeIdx * nBinPackAreasPerPlate;
            int endAreaIdx = std::min((planeIdx + 1) * nBinPackAreasPerPlate, nConnectedAreas) - 1;
            std::vector<int> areaIdxs(endAreaIdx - startAreaIdx + 1);
            std::iota(areaIdxs.begin(), areaIdxs.end(), startAreaIdx);
            vecAreaIdxs.push_back(areaIdxs);
            // double totalArea = 0.0;
            // for (const int i: areaIdxs) totalArea += areas[i];
        }
    }
    else if (binPackType == BinPackTypeEnum::FIX_SCALE)
    {
        std::vector<size_t> sortedAreaIdxs = NS_Utils::getSortedIdxes(areas);

        double areaLimit = std::pow(BIN_PACK_UNITS_PER_INT * BIN_PACK_BOX_LENGTH, 2);

        // std::vector<std::vector<int>> vecAreaIdxs;
        // std::vector<double> vecTotalAreas;
        int curIdx = 0;
        while (curIdx < nConnectedAreas) {
            double totalArea = 0.0;
            std::vector<int> areaIdxs;
            while ((areaIdxs.size() == 0) || ((curIdx < nConnectedAreas) && (totalArea + areas[sortedAreaIdxs[curIdx]] < areaLimit))) {
                areaIdxs.push_back(sortedAreaIdxs[curIdx]);
                totalArea += areas[sortedAreaIdxs[curIdx]];
                curIdx++;
            }
            vecAreaIdxs.push_back(areaIdxs);
            // vecTotalAreas.push_back(totalArea);
        }

        data.maxPlaneIdx = vecAreaIdxs.size() - 1;
    }
    else if (binPackType == BinPackTypeEnum::SORTED_FIX_NUM)
    {
        std::vector<size_t> sortedAreaIdxs = NS_Utils::getSortedIdxes(areas);

        data.maxPlaneIdx = (int)std::ceil((float)nConnectedAreas / nBinPackAreasPerPlate) - 1;
    
        for (int planeIdx = 0; planeIdx <= data.maxPlaneIdx; planeIdx++) {
            int startAreaIdx = planeIdx * nBinPackAreasPerPlate;
            int endAreaIdx = std::min((planeIdx + 1) * nBinPackAreasPerPlate, nConnectedAreas) - 1;
            std::vector<int> areaIdxs;
            // double totalArea = 0.0;
            for (int i = startAreaIdx; i <= endAreaIdx; i++) {
                areaIdxs.push_back(sortedAreaIdxs[i]);
                // totalArea += areas[sortedAreaIdxs[i]];
            }
            vecAreaIdxs.push_back(areaIdxs);
        }
    }
    else if (binPackType == BinPackTypeEnum::FACE_AREA_ADAPTIVE)
    {
        float mtlArea = (float)mtlHeight * (float)mtlWidth;

        std::vector<size_t> sortedAreaIdxs = NS_Utils::getSortedIdxes(areas);

        int curIdx = 0;
        while (curIdx < nConnectedAreas) {
            double totalArea = 0.0;
            std::vector<int> areaIdxs;

            // no sort
            while ((areaIdxs.size() == 0) || ((curIdx < nConnectedAreas) && (totalArea + areas[curIdx] < adapatThreshold * mtlArea))) {
                areaIdxs.push_back(curIdx);
                totalArea += areas[curIdx];
                curIdx++;
            }

            vecAreaIdxs.push_back(areaIdxs);
        }

        data.maxPlaneIdx = vecAreaIdxs.size() - 1;
    
        std::cout << prompt << "#plates:" << vecAreaIdxs.size() << std::endl;
        std::cout << prompt << "#faces per plate:";
        for (auto it: vecAreaIdxs) std::cout << " " << it.size();
        std::cout << std::endl;
    }

    assert(vecAreaIdxs.size() == data.maxPlaneIdx + 1);
    
    if (flagConformalMap) {
        for (int planeIdx = 0; planeIdx <= data.maxPlaneIdx; planeIdx++) {
            BinPackTextureTriangles2DSinglePlane(flagConformalMap, planeIdx, vecAreaIdxs[planeIdx], areas, minBounds, maxBounds, visitedUVPts, addUVCnts);
            std::cout << prompt << planeIdx + 1 << "/" << data.maxPlaneIdx + 1 << " bin packing completed" << std::endl;
        }
    } else {
#pragma omp parallel for
        for (int planeIdx = 0; planeIdx <= data.maxPlaneIdx; planeIdx++) {
            BinPackTextureTriangles2DSinglePlane(flagConformalMap, planeIdx, vecAreaIdxs[planeIdx], areas, minBounds, maxBounds, visitedUVPts, addUVCnts);
            std::cout << prompt << planeIdx + 1 << "/" << data.maxPlaneIdx + 1 << " bin packing completed" << std::endl;
        }
    }

    if (flagConformalMap) {
        // update texture triangle vertex index since we add some new UV coordiantes
        unsigned int cumAddUVCnt = 0;
        data.uvCntsAfterRePack.resize(data.uvCnts.size(), 0);

        for (int k = 1; k < data.textureTriangles.size(); k++) {
            cumAddUVCnt += addUVCnts[k - 1];
            data.uvCntsAfterRePack[k] = data.uvCnts[k] + cumAddUVCnt;
    
            for (int f = 0; f < data.textureTriangles[k].size(); f++) {
                for (int v = 0; v < data.textureTriangles[k][f].length; v++) {
                    data.textureTriangles[k][f].c[v] += cumAddUVCnt;
                }
            }
        }
    }

    if (globalDebugFlag == 1) {
        std::cout << prompt + "[Debug] newly added #uv from re-packing: ";
        for (auto it: addUVCnts) std::cout << it << " ";
        std::cout << std::endl;
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << prompt + "... complete re-packing compact material image in " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms" << std::endl;
}

void OurMeshData::BinPackTextureTriangles2DSinglePlane(const bool& flagConformalMap, const int& planeIdx, // const double& totalArea,
                                                       const std::vector<int>& areaIdxs, const std::vector<double>& areas,
                                                       const std::vector<bff::Vector>& minBounds, const std::vector<bff::Vector>& maxBounds,
                                                       std::vector<std::vector<int>>& visitedUVPts, std::vector<unsigned int>& addUVCnts)
{
    int nConnectedAreas = areaIdxs.size();

    // NOTE: we compute total area in each plane separately,
    // the scale/unitsPerInt will be different across planes.
    double totalArea = 0.0;
    for (const int& i: areaIdxs) totalArea += areas[i];

    // quantize boxes
    std::vector<bff::Vector> originalCenters(nConnectedAreas);
    std::vector<rbp::Rect> rectangles(nConnectedAreas);
    int minBoxLength = BIN_PACK_BOX_LENGTH;
    int maxBoxLength = BIN_PACK_BOX_LENGTH;
    // we need this value to convert uv \in [0, 1] to interger value and then convert them back
    double unitsPerInt = sqrt(totalArea) / (double)maxBoxLength;

    for (int i = 0; i < nConnectedAreas; i++) {
        int minX = static_cast<int>(floor(minBounds[areaIdxs[i]].x / unitsPerInt));
        int minY = static_cast<int>(floor(minBounds[areaIdxs[i]].y / unitsPerInt));
        int maxX = static_cast<int>(ceil(maxBounds[areaIdxs[i]].x / unitsPerInt));
        int maxY = static_cast<int>(ceil(maxBounds[areaIdxs[i]].y / unitsPerInt));

        int width = maxX - minX;
        int height = maxY - minY;
        rectangles[i] = rbp::Rect{minX, minY, width, height};
        originalCenters[i].x = (minX + maxX) / 2.0;
        originalCenters[i].y = (minY + maxY) / 2.0;
        originalCenters[i] *= unitsPerInt;
    }

    // pack rectangles
    std::vector<bool> flippedBins(nConnectedAreas);
    std::vector<bff::Vector> newCenters(nConnectedAreas);
    bff::Vector modelMinBounds;
    bff::Vector modelMaxBounds;

    int iter = 0;

    do {
        if (bff::BinPacking::attemptPacking(maxBoxLength, unitsPerInt, rectangles, newCenters,
                                            flippedBins, modelMinBounds, modelMaxBounds)) break;

        minBoxLength = maxBoxLength;
        maxBoxLength = static_cast<int>(ceil(minBoxLength*1.2));
        iter++;
    } while (iter < 50);

    if (iter < 50) {
        // binary search on box length
        minBoxLength = 5000;
        maxBoxLength += 1;

        while (minBoxLength <= maxBoxLength) {
            int boxLength = (minBoxLength + maxBoxLength) / 2;
            if (boxLength == minBoxLength) break;

            if (bff::BinPacking::attemptPacking(boxLength, unitsPerInt, rectangles, newCenters,
                                                flippedBins, modelMinBounds, modelMaxBounds)) {
                maxBoxLength = boxLength;

            } else {
                minBoxLength = boxLength;
            }
        }

        bff::BinPacking::attemptPacking(maxBoxLength, unitsPerInt, rectangles, newCenters,
                                        flippedBins, modelMinBounds, modelMaxBounds);
    }

    modelMinBounds *= unitsPerInt;
    modelMaxBounds *= unitsPerInt;

    // compute uv radius and shift
    bff::Vector minExtent(modelMinBounds.x, modelMinBounds.y);
    double dx = modelMaxBounds.x - minExtent.x;
    double dy = modelMaxBounds.y - minExtent.y;
    double extent = std::max(dx, dy);
    minExtent.x -= (extent - dx) / 2.0;
    minExtent.y -= (extent - dy) / 2.0;

    if (flagConformalMap) {
        BinPackPostprocessForConformalMap(planeIdx, nConnectedAreas, areaIdxs, minExtent, extent, flippedBins,
                                          originalCenters, newCenters, visitedUVPts, addUVCnts);
    } else {
        BinPackPostprocessForNonConformalMap(planeIdx, nConnectedAreas, areaIdxs, minExtent, extent, flippedBins,
                                             originalCenters, newCenters);
    }
}


void OurMeshData::BinPackPreprocessForConformalMap(std::vector<double>& areas,
                                                   std::vector<bff::Vector>& minBounds, std::vector<bff::Vector>& maxBounds,
                                                   const int& globalDebugFlag, const std::string& prompt)
{
    ComputeTriangleAdjacency2D(globalDebugFlag, prompt);

    data.infoTextureTriangleAreaIdx.resize(data.textureTriangles.size(), std::vector<unsigned int>());
    for (int k = 0; k < data.textureTriangles.size(); k++) {
        data.infoTextureTriangleAreaIdx[k].resize(data.textureTriangles[k].size(), 0);
    }

    auto searchStart = std::chrono::high_resolution_clock::now();

    int nConnectedAreas = 0;
    std::vector<bool> visited(data.nTextureTriangles, false);

    // for (int p = 0; p <= data.maxPlaneIdx; p++) {
    for (int k = 0; k < data.textureTriangles.size(); k++) {
        for (int f = 0; f < data.textureTriangles[k].size(); f++) {
            if (!visited[data.triangleGlobalIdx[k][f]]) {
                std::vector<NS_Utils::Tuple> tmpSet;
                SearchConnectedArea(data.planeIdxAfterMP[data.triangleGlobalIdx[k][f]], k, f, nConnectedAreas, tmpSet, visited, minBounds, maxBounds);
                data.infoAreaTriangleIdxSet.push_back(tmpSet);

                // totalArea += (maxBounds[nConnectedAreas].x - minBounds[nConnectedAreas].x) * (maxBounds[nConnectedAreas].y - minBounds[nConnectedAreas].y);
                areas.push_back((maxBounds[nConnectedAreas].x - minBounds[nConnectedAreas].x) * (maxBounds[nConnectedAreas].y - minBounds[nConnectedAreas].y));

                nConnectedAreas++;
            }
        }
    }
    // }

    auto searchFinish = std::chrono::high_resolution_clock::now();
    std::cout << prompt + "[conformal mapping] complete searching " << nConnectedAreas << " connected areas from " << data.nPointTriangles << " faces in " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(searchFinish - searchStart).count() << " ms" << std::endl;
}


void OurMeshData::BinPackPreprocessForNonConformalMap(std::vector<double>& areas,
                                                      std::vector<bff::Vector>& minBounds, std::vector<bff::Vector>& maxBounds,
                                                      const bool& flagAreaMesh, const NS_StreamFile::StreamFile& streamInfos,
                                                      const Eigen::MatrixXf& floatNDC, const std::vector<Eigen::MatrixXf>& faceAreaMatrix,
                                                      const int& globalDebugFlag, const std::string& prompt)
{
    auto start = std::chrono::high_resolution_clock::now();

    int nConnectedAreas = data.nPointTriangles;

    data.uvPtsAfterRePack.resize(data.pointTriangles.size());
    for (int k = 0; k < data.pointTriangles.size(); k++) data.uvPtsAfterRePack[k].resize(3 * data.pointTriangles[k].size());

    data.uvCntsAfterRePack.push_back(0);
    for (int k = 1; k < data.uvPtsAfterRePack.size(); k++) {
        data.uvCntsAfterRePack.push_back(data.uvCntsAfterRePack[k - 1] + data.uvPtsAfterRePack[k - 1].size());
    }

    data.nTextureTriangles = data.nPointTriangles;
    data.textureTriangles.resize(data.pointTriangles.size());
    for (int k = 0; k < data.textureTriangles.size(); k++) {
        for (int f = 0; f < data.pointTriangles[k].size(); f++) {
            NS_Utils::Triple tT;
            tT.c[0] = data.uvCntsAfterRePack[k] + 3 * f;
            tT.c[1] = data.uvCntsAfterRePack[k] + 3 * f + 1;
            tT.c[2] = data.uvCntsAfterRePack[k] + 3 * f + 2;
            data.textureTriangles[k].push_back(tT);
        }
    }

    // create a map from global index to a pair of (modelIdx, faceIdx)
    data.triangleGlobalIdxToPairIdx.resize(data.nTextureTriangles, std::pair<unsigned int, unsigned int>());
    for (int k = 0; k < data.textureTriangles.size(); k++) {
        for (int f = 0; f < data.textureTriangles[k].size(); f++) {
            data.triangleGlobalIdxToPairIdx[data.triangleGlobalIdx[k][f]] = std::make_pair(k, f);
        }
    }

    areas.resize(nConnectedAreas);
    minBounds.resize(nConnectedAreas);
    maxBounds.resize(nConnectedAreas);

    float rgbHeight, rgbWidth;
    int maxAreaViewIdx = 0;
    Eigen::MatrixXd xcoords, ycoords;
    Eigen::Matrix3d faceUVs;
    Eigen::Vector3d faceUV1, faceUV2, faceUV3;

    Eigen::MatrixXd NDC = floatNDC.cast<double>();

    for (int k = 0; k < data.textureTriangles.size(); k++) {

        // [#camears, #faces]
        // We have made sure that non-valid face-camera pair has area of zero.
        Eigen::MatrixXf faceMaxAreas = faceAreaMatrix[k].colwise().maxCoeff();

        for (int f = 0; f < data.textureTriangles[k].size(); f++) {

            Eigen::Vector2d tmpMinBound, tmpMaxBound;

            if (flagAreaMesh)
            {
                NS_Utils::UVCoordsFromTriangle3D(data.Points(Eigen::all, data.pointTriangles[k][f].c[0]),
                                                 data.Points(Eigen::all, data.pointTriangles[k][f].c[1]),
                                                 data.Points(Eigen::all, data.pointTriangles[k][f].c[2]),
                                                 data.uvPtsAfterRePack[k][3 * f],
                                                 data.uvPtsAfterRePack[k][3 * f + 1],
                                                 data.uvPtsAfterRePack[k][3 * f + 2],
                                                 tmpMinBound, tmpMaxBound);
            }
            else
            {
                for (int m = 0; m < streamInfos.NumberOfEntries(); m++) {
                    if (faceAreaMatrix[k](m, f) == faceMaxAreas(0, f)) {
                        maxAreaViewIdx = m;
                        break;
                    }
                }
                assert (maxAreaViewIdx < streamInfos.NumberOfEntries());

                // std::cout << k << " " << f << " " << maxAreaViewIdx << " " << faceMaxAreas(0, f) << std::endl; // << " " << faceAreaMatrix[k](Eigen::all, f) << std::endl;

                streamInfos.RGBSizeQuery(maxAreaViewIdx, rgbHeight, rgbWidth);
                xcoords = NDC(2 * maxAreaViewIdx, data.pointTriangles[k][f].c) * rgbHeight;  
                ycoords = NDC(2 * maxAreaViewIdx + 1, data.pointTriangles[k][f].c) * rgbWidth;
                faceUVs << xcoords, ycoords, Eigen::Matrix<double, 1, 3>::Zero();

                // std::cout << "Bin pack: " << xcoords << " " << ycoords << " " << std::endl;
                // std::cout << faceUVs << std::endl;
                // exit(EXIT_FAILURE);

                NS_Utils::UVCoordsFromTriangle3D(faceUVs(Eigen::all, 0),
                                                 faceUVs(Eigen::all, 1),
                                                 faceUVs(Eigen::all, 2),
                                                 data.uvPtsAfterRePack[k][3 * f],
                                                 data.uvPtsAfterRePack[k][3 * f + 1],
                                                 data.uvPtsAfterRePack[k][3 * f + 2],
                                                 tmpMinBound, tmpMaxBound);
            }
            
            minBounds[data.triangleGlobalIdx[k][f]] = bff::Vector(tmpMinBound(0, 0), tmpMinBound(1, 0));
            maxBounds[data.triangleGlobalIdx[k][f]] = bff::Vector(tmpMaxBound(0, 0), tmpMaxBound(1, 0));

            areas[data.triangleGlobalIdx[k][f]] = (maxBounds[data.triangleGlobalIdx[k][f]].x - minBounds[data.triangleGlobalIdx[k][f]].x) * \
                                                  (maxBounds[data.triangleGlobalIdx[k][f]].y - minBounds[data.triangleGlobalIdx[k][f]].y);
        }
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << prompt + "[non-conformal mapping] complete preprocessing " << nConnectedAreas << " connected areas from " << data.nPointTriangles << " faces in " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms" << std::endl;
}


void OurMeshData::BinPackPostprocessForConformalMap(const int& planeIdx, const int& nConnectedAreas, const std::vector<int>& areaIdxs,
                                                    const bff::Vector& minExtent, const double& extent, const std::vector<bool>& flippedBins,
                                                    const std::vector<bff::Vector>& originalCenters, const std::vector<bff::Vector>& newCenters,
                                                    std::vector<std::vector<int>>& visitedUVPts, std::vector<unsigned int>& addUVCnts)
{
    for (int i = 0; i < nConnectedAreas; i++) {

        for (const NS_Utils::Tuple it: data.infoAreaTriangleIdxSet[areaIdxs[i]]) { 
            int k = it.k;
            int f = it.f;

            data.planeIdxAfterRePack[data.triangleGlobalIdx[k][f]] = planeIdx;

            for (int v = 0; v < data.textureTriangles[k][f].length; v++) {
                // - no mappedToSphere
                // - radius is 1 since we directly use uvPts
                // - we need normalize
                bff::Vector tmpOldUV(data.uvPts[k][data.textureTriangles[k][f].c[v] - data.uvCnts[k]](0),
                                     data.uvPts[k][data.textureTriangles[k][f].c[v] - data.uvCnts[k]](1));
                Eigen::Vector2d tmpNewUV = \
                    getUV(tmpOldUV, false, 1.0, 1.0, originalCenters[i], newCenters[i],
                          minExtent, extent, flippedBins[i], true);

                if (visitedUVPts[k][data.textureTriangles[k][f].c[v] - data.uvCnts[k]] == -1)
                {
                    // this uv point has never been filled before
                    data.uvPtsAfterRePack[k][data.textureTriangles[k][f].c[v] - data.uvCnts[k]] = tmpNewUV;
                    visitedUVPts[k][data.textureTriangles[k][f].c[v] - data.uvCnts[k]] = planeIdx;
                }
                else if (visitedUVPts[k][data.textureTriangles[k][f].c[v] - data.uvCnts[k]] != planeIdx ||
                         (tmpNewUV - data.uvPtsAfterRePack[k][data.textureTriangles[k][f].c[v] - data.uvCnts[k]]).norm() > FLOAT_PRECISION)
                {
                    // we need to add new texture vertex
                    data.uvPtsAfterRePack[k].push_back(tmpNewUV);
                    data.textureTriangles[k][f].c[v] = data.uvCnts[k] + data.uvPtsAfterRePack[k].size() - 1;
                    addUVCnts[k]++;
                }
            }
        }
    }
}


void OurMeshData::BinPackPostprocessForNonConformalMap(const int& planeIdx, const int& nConnectedAreas, const std::vector<int>& areaIdxs,
                                                    const bff::Vector& minExtent, const double& extent, const std::vector<bool>& flippedBins,
                                                    const std::vector<bff::Vector>& originalCenters, const std::vector<bff::Vector>& newCenters)
{
    for (int i = 0; i < nConnectedAreas; i++) {
        int k = data.triangleGlobalIdxToPairIdx[areaIdxs[i]].first;
        int f = data.triangleGlobalIdxToPairIdx[areaIdxs[i]].second;
        
        data.planeIdxAfterRePack[areaIdxs[i]] = planeIdx;
          
        for (int v = 0; v < data.textureTriangles[k][f].length; v++) {
            // - no mappedToSphere
            // - radius is 1 since we directly use uvPts
            // - we need normalize
            bff::Vector tmpOldUV(data.uvPtsAfterRePack[k][data.textureTriangles[k][f].c[v] - data.uvCntsAfterRePack[k]](0),
                                 data.uvPtsAfterRePack[k][data.textureTriangles[k][f].c[v] - data.uvCntsAfterRePack[k]](1));
            Eigen::Vector2d tmpNewUV = \
                getUV(tmpOldUV, false, 1.0, 1.0, originalCenters[i], newCenters[i], minExtent, extent, flippedBins[i], true);
           
            data.uvPtsAfterRePack[k][data.textureTriangles[k][f].c[v] - data.uvCntsAfterRePack[k]] = tmpNewUV;
        }
    }
}


void OurMeshData::SearchConnectedArea(const int& planeIdx, const int& modelIdx, const int& faceIdx, const int& connectedAreaIdx,
                                      std::vector<NS_Utils::Tuple>& idxSet, std::vector<bool>& visited,
                                      std::vector<bff::Vector>& minBounds, std::vector<bff::Vector>& maxBounds)
{
    bff::Vector tmpMinBounds(std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
    bff::Vector tmpMaxBounds(std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest());

    std::queue<NS_Utils::Tuple> q;
    q.push(NS_Utils::Tuple{(unsigned int)modelIdx, (unsigned int)faceIdx});
    visited[data.triangleGlobalIdx[modelIdx][faceIdx]] = true;

    unsigned int cnt = 0;
    while (!q.empty()) {
        NS_Utils::Tuple tmp = q.front();
        q.pop();

        data.infoTextureTriangleAreaIdx[tmp.k][tmp.f] = connectedAreaIdx;

        idxSet.push_back(tmp);

        // find bound value for this connected area
        GetBounds(tmp.k, tmp.f, tmpMinBounds, tmpMaxBounds);

        for (auto it: data.infoTextureTriangleAdjacency2D[tmp.k][tmp.f]) {
            if ((data.planeIdxAfterMP[data.triangleGlobalIdx[tmp.k][it]] == planeIdx) && !visited[data.triangleGlobalIdx[tmp.k][it]]) {
                visited[data.triangleGlobalIdx[tmp.k][it]] = true;
                q.push(NS_Utils::Tuple{tmp.k, it});
            }
        }
    }

    minBounds.push_back(tmpMinBounds);
    maxBounds.push_back(tmpMaxBounds);
}


void OurMeshData::GetBounds(const int& modelIdx, const int& faceIdx, bff::Vector& minBounds, bff::Vector& maxBounds)
{
    for (int i = 0; i < data.textureTriangles[modelIdx][faceIdx].length; i++) {
        // first element of uvPts is for horizontal
        minBounds.x = std::min(data.uvPts[modelIdx][data.textureTriangles[modelIdx][faceIdx].c[i] - data.uvCnts[modelIdx]](0), minBounds.x);
        minBounds.y = std::min(data.uvPts[modelIdx][data.textureTriangles[modelIdx][faceIdx].c[i] - data.uvCnts[modelIdx]](1), minBounds.y);
        maxBounds.x = std::max(data.uvPts[modelIdx][data.textureTriangles[modelIdx][faceIdx].c[i] - data.uvCnts[modelIdx]](0), maxBounds.x);
        maxBounds.y = std::max(data.uvPts[modelIdx][data.textureTriangles[modelIdx][faceIdx].c[i] - data.uvCnts[modelIdx]](1), maxBounds.y);
    }
}

} /* end namespace */