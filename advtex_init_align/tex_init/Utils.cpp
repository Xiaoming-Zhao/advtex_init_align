#include <iomanip>
#include <iostream>
#include <string.h>
#include <cstdlib>

#include "Utils.h"

namespace NS_Utils {

bool CheckLineSegIntersect2D(const float& x1, const float& y1,
                             const float& x2, const float& y2,
                             const float& x3, const float& y3,
                             const float& x4, const float& y4)
{
    // https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    // line seg 1: (x1, y1), (x2, y2)
    // line seg 2: (x3, y3), (x4, y4)

    // check singularity
    float determinant = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

    if (std::abs(determinant) < FLOAT_PRECISION) return false;

    float l1_coeff = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / determinant;
    float l2_coeff = - ((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / determinant;

    // Pay attention to precision of floating number
    return (l1_coeff > FLOAT_PRECISION) && (l1_coeff < 1 - FLOAT_PRECISION) && (l2_coeff > FLOAT_PRECISION) && (l2_coeff < 1 - FLOAT_PRECISION);
}

bool GetTwoCommonElementsInTriple(const Triple& triple1, const Triple& triple2, int& elem1, int& elem2)
{
    for (int i = 0; i < nCombine; i++) {
        if ((triple1.c[combination[i][0]] == triple2.c[combination[i][2]]) &&
            (triple1.c[combination[i][1]] == triple2.c[combination[i][3]])) {
            elem1 = triple1.c[combination[i][0]];
            elem2 = triple1.c[combination[i][1]];
            return true;
        }
    }

    return false;
}

void UVCoordsFromTriangle3D(const Eigen::Vector3d& pt1, const Eigen::Vector3d pt2, const Eigen::Vector3d pt3,
                            Eigen::Vector2d& uv1, Eigen::Vector2d& uv2, Eigen::Vector2d& uv3,
                            Eigen::Vector2d& minUV, Eigen::Vector2d& maxUV)
{
    // This function outputs the 2D uv coordinates from a 3D triangle face.
    // +U: right, +V: down
    // 1 |\
    //   | \
    //   |  \ 3
    //   |  /
    //   | /
    // 2 |/
    // It maps pt1 to uv1 (0, 0), pt2 to uv2 (0, |pt1 - pt2|), where |pt1 - pt2| is the length of vector.
    // uv3 is computed as:
    // - v3 = |pt3 - pt1| cos(angle between pt3 - pt1, pt2 - pt1)
    // - u3 = |pt3 - pt1| |sin(angle between pt3 - pt1, pt2 - pt1)|
    //
    // after get all uv, check whether exists uv < 0.
    // Translate so that all uv are non-negative.

    Eigen::Matrix<double, 2, 3> allUVs = Eigen::Matrix<double, 2, 3>::Zero();

    double normPt1ToPt2 = (pt2 - pt1).norm();

    // for v2
    allUVs(1, 1) = normPt1ToPt2;

    if (normPt1ToPt2 != 0) {
        // this triangle is not degraded
        // for u3
        allUVs(0, 2) = (pt2 - pt1).cross(pt3 - pt1).norm() / normPt1ToPt2;
        // for v3
        allUVs(1, 2) = (pt3 - pt1).dot(pt2 - pt1) / normPt1ToPt2;
    }

    minUV = allUVs.rowwise().minCoeff();

    // all points's u coordinate should all be non-negative.
    assert (minUV(0, 0) >= 0);
    for (int i = 0; i < 3; i++) {
        allUVs(0, i) -= minUV(0, 0);
        allUVs(1, i) -= minUV(1, 0);
    }

    // update boundary coordinate info
    minUV = allUVs.rowwise().minCoeff();
    maxUV = allUVs.rowwise().maxCoeff();

    assert ((allUVs.array() >= 0.0).all());

    uv1 = allUVs(Eigen::all, 0);
    uv2 = allUVs(Eigen::all, 1);
    uv3 = allUVs(Eigen::all, 2);
}

int ArgPos(char *str, int argc, char **argv) {
    int pos = 0;
    for (pos = 1; pos < argc; pos++) {
        if (!strcmp(str, argv[pos])) {
            if (pos == argc - 1) {
                std::cout << "Argument is missing for " << str << std::endl;
            }
            return pos;
        }
    }
    return -1;
}

void MakeDir(const std::string dir) {
    std::string cmd = "mkdir -p " + dir;
    int status = system(cmd.c_str());
    if (status < 0)
        {std::cout << "Error: " << strerror(errno) << '\n';}
    else
    {
        if (WIFEXITED(status))
            std::cout << "Make directory " + dir + " normally, exit code " << WEXITSTATUS(status) << std::endl;
        else {
            std::cout << "Unable to make directory: " + dir << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}


void WriteMatrixBin(const Eigen::MatrixXf& Mat, std::ofstream& outFileStream, const bool& orderC)
{
    unsigned int rows = Mat.rows(), cols = Mat.cols();
    assert (std::numeric_limits<unsigned int>::max() > rows);
    assert (std::numeric_limits<unsigned int>::max() > cols);

    if (orderC)
    {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                outFileStream.write((char*)&Mat(i, j), sizeof(float));
            }
        }
    } else {
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                outFileStream.write((char*)&Mat(i, j), sizeof(float));
            }
        }
    }
}


void PlotNDC(boost::gil::rgb8_image_t* img, Eigen::MatrixXf& NDC, const int& ID) {

    auto view = boost::gil::view(*img);

    for(int k=0;k<NDC.cols();++k) {
        float ptx = NDC(2*ID, k)*view.height();
        float pty = NDC(2*ID+1, k)*view.width();

        for(int u=-1;u<=1;++u) {
            for(int v=-1;v<=1;++v) {
                if(pty-u>=0 && pty-u<view.width() && ptx-v>=0 && ptx-v<=view.height()) {
                    view(pty-u, ptx-v) = boost::gil::rgb8_pixel_t{0, 0, 255};
                }
            }
        }
    }
    boost::gil::write_view("NDCPoints.png", boost::gil::const_view(*img), boost::gil::png_tag());
}

void TestMPSolver(NS_MPSolver::MPSolver<float, int>& MP, std::default_random_engine& generator, std::uniform_real_distribution<float>& uniformDist)
{
    // MP.CreateGraph(2, 2);
    // float tmpUnary[4] = {1, 0, 0.5, 0};
    // std::vector<float> tmpPairPotentials1;
    // MP.UpdateGraph(MP.GraphUpdateTypes::OnlyUnary, tmpUnary, 0, NULL, tmpPairPotentials1);
    // std::vector<float> tmpPairPotentials2{0.0, 100.0, 100.0, 0.0};
    // float tmpPairInfos1[6] = {1.0, 2.0, 1.0};
    // MP.UpdateGraph(MP.GraphUpdateTypes::OnlyPair, NULL, 1, tmpPairInfos1, tmpPairPotentials2);
    
    // float tmpUnary[8] = {};
    // // face1, face2 and face4 prefer to stay in plane 1 since they do not overlap with others
    // tmpUnary[0] = uniformDist(generator);
    // tmpUnary[1] = 1 - tmpUnary[0];
    // tmpUnary[2] = uniformDist(generator);
    // tmpUnary[3] = 1 - tmpUnary[2];
    // tmpUnary[6] = uniformDist(generator);
    // tmpUnary[7] = 1 - tmpUnary[6];
    // // face3 prefers to stay in plane 2 since it overlaps with plane 1
    // tmpUnary[5] = uniformDist(generator);
    // tmpUnary[4] = 1 - tmpUnary[5];

    // std::cout << "Unary potential: ";
    // for (int i = 0; i < 8; i++) std::cout << tmpUnary[i] << " ";
    // std::cout << std::endl;

    // four faces, two planes
    // face3 overlaps with face1; face1 and face2 are adjacent; face3 and face4 are adjacent
    MP.CreateGraph(4, 2);

    // face1, face2 and face4 prefer to stay in plane 1 since they do not overlap with others
    std::vector<float> tmpUnary1{0.5, 0, 0.5, 0, 0, 0.5, 0.5, 0};
    std::vector<float> tmpPairInfos1;
    std::vector<float> tmpPairPotentials1;
    MP.UpdateGraph(MP.GraphUpdateTypes::OnlyUnary, tmpUnary1, tmpPairInfos1, tmpPairPotentials1);

    // (face1, face2) and (face3 and face4) are adjacent, note, regionIDs are 1-based
    std::vector<float> tmpUnary2;
    std::vector<float> tmpPairPotentials2{10.0, 0.0, 0.0, 10.0};
    std::vector<float> tmpPairInfos2{0.0, 1.0, 1.0, 2.0, 3.0, 1.0};
    MP.UpdateGraph(MP.GraphUpdateTypes::OnlyPair, tmpUnary2, tmpPairInfos2, tmpPairPotentials2);

    // face1 and face3 overlap
    std::vector<float> tmpPairPotentials3{0.0, 10.0, 10.0, 0.0};
    std::vector<float> tmpPairInfos3{0.0, 2.0, 1.0};
    MP.UpdateGraph(MP.GraphUpdateTypes::OnlyPair, tmpUnary2, tmpPairInfos3, tmpPairPotentials3);

    // face1 and face2 are on one plane, face3 and face4 locate in the other one.
    MP.Solve(10);

    for (int k = 0; k < MP.nGraphVars; k++) {
        std::cout << k << " "; 
        for (int i = 0; i < MP.graphVarCardinality; i++) {
            std::cout << MP.varBeliefs[k * MP.graphVarCardinality + i] << " ";
        }
        std::cout << std::endl;
    }

    for(int k = 0; k < MP.nGraphVars; k++) {
        std::cout << "var " << k + 1 << ": " << MP.varState[k]  << std::endl;
    }

    MP.DestroyGraph();
}

} /* end namespace NS_Utils */