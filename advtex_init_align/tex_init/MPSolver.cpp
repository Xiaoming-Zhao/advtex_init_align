#include <iostream>
#include <fstream>

#include "MPSolver.h"

namespace NS_MPSolver {

template <typename T, typename S>
MPSolver<T, S>::MPSolver() {}

template <typename T, typename S>
MPSolver<T, S>::~MPSolver()
{
    DestroyGraph();
}

template <typename T, typename S>
void MPSolver<T, S>::DestroyGraph()
{
    std::vector<typename MPGraph<T, S>::PotentialID>().swap(potentialIDs);
    std::vector<typename MPGraph<T, S>::RegionID>().swap(regionIDs);
    std::vector<S>().swap(varState);
    std::vector<T>().swap(varBeliefs);
    std::vector<T>().swap(maxBelief);
    graph.reset(nullptr);
}

template <typename T, typename S>
bool MPSolver<T, S>::CreateGraph(const S& numVars, const S& cardinality)
{
    nGraphVars = numVars;
    graphVarCardinality = cardinality;

    // I think we are good no matter whether we use () or not
    // since MPGraph has user-defined constructor
    graph = std::unique_ptr<MPGraph<T, S>>(new MPGraph<T, S>());

    // Just a sanity check for calling "new" correctly
    assert(graph->entropy == 0);
    assert(graph->region_belief_sums == nullptr);

    graph->AddVariables(std::vector<S>(nGraphVars, graphVarCardinality));

    return true;
}

template <typename T, typename S>
bool MPSolver<T, S>::UpdateGraph(const int& updateType, std::vector<T>& unaryInfos,
                                 const std::vector<T>& pairInfos, std::vector<T>& pairPotentials)
{
    // We enable an indicator for caller to specify the update type,
    // then it is the caller's responsibilities to make sure the given information is correct.
    // [deprecated] Several update scenarios:
    // [deprecated] - only update unary information: unaryInfos != NULL, nPairs == 0, pairInfos == NULL, pairPotentials.size() == 0
    // [deprecated] - only update pairwise information: unaryInfors is NULL, pairInfos.size() == graphVarCardinality^2
    // [deprecated] - update both unary and pairwise: all arguments are not NULL or has zero size

    if ((updateType == GraphUpdateTypes::OnlyUnary) || (updateType == GraphUpdateTypes::BothUnaryPair)) {
        // T* unaryPtr = &unaryInfos[0];
        T* unaryPtr = &unaryInfos[0];
        for(int k = 0; k < nGraphVars; ++k, unaryPtr += graphVarCardinality) {
            potentialIDs.push_back(graph->AddPotential(typename MPGraph<T, S>::PotentialVector(unaryPtr, graphVarCardinality)));
        }
    }

    if ((updateType == GraphUpdateTypes::OnlyPair) || (updateType == GraphUpdateTypes::BothUnaryPair)) {
        assert(pairPotentials.size() == graphVarCardinality * graphVarCardinality);  
        potentialIDs.push_back(graph->AddPotential(typename MPGraph<T, S>::PotentialVector(&pairPotentials[0], pairPotentials.size())));
    }

    if ((updateType == GraphUpdateTypes::OnlyUnary) || (updateType == GraphUpdateTypes::BothUnaryPair)) {
        for(int k = 0; k < nGraphVars; ++k) {
            regionIDs.push_back(graph->AddRegion(1.0, 1.0, std::vector<S>{k}, potentialIDs[k]));
        }
    }

    if ((updateType == GraphUpdateTypes::OnlyPair) || (updateType == GraphUpdateTypes::BothUnaryPair)) {
        assert (pairInfos.size() % 3 == 0);
        int nPairs = pairInfos.size() / 3;
        for(int k = 0; k < nPairs; k++) {
            S node1 = S(pairInfos[3 * k]);
            S node2 = S(pairInfos[3 * k + 1]);
            T potential_weight = pairInfos[3 * k + 2];
            regionIDs.push_back(graph->AddRegion(1.0, potential_weight, std::vector<S>{node1, node2}, potentialIDs.back()));
            graph->AddConnection(regionIDs[node1], regionIDs.back());
            graph->AddConnection(regionIDs[node2], regionIDs.back());
        }
    }

    return true;
}

template <typename T, typename S>
bool MPSolver<T, S>::Solve(const int& nIters, const bool& flagSaveFullBeliefs, const std::string& fileAllBeliefs) {

    graph->AllocateMessageMemory();

    std::cout << "MP graph #regions: " << graph->GetGraphSize() << std::endl;
    std::cout << "MP graph #regions with parents: " << graph->NumberOfRegionsWithParents() << std::endl;

    std::vector<T> lambda(graph->GetLambdaSize());
    std::cout << "MP graph lambda size: " << lambda.size() << std::endl;

    T epsilon = 0;
    bool onlyUnaries = true;

    typename MPGraph<T, S>::DualWorkspaceID dw = graph->AllocateDualWorkspaceMem(epsilon);
    typename MPGraph<T, S>::RRegionWorkspaceID rrw = graph->AllocateReparameterizeRegionWorkspaceMem(epsilon);

    CPrecisionTimer CTmr;
    CTmr.Start();
    RMP<T, S> RMPAlgo(*graph);
    float dual = RMPAlgo.RunMP(&lambda[0], epsilon, nIters, dw, rrw);
    std::cout << "Time: " << CTmr.Stop() << "; Dual: " << dual << std::endl;

    std::vector<T> beliefs(graph->ComputeBeliefSize(onlyUnaries), T(0));
    graph->ComputeBeliefs(&lambda[0], epsilon, &beliefs[0], onlyUnaries);

    graph->DeAllocateDualWorkspaceMem(dw);
    graph->DeAllocateReparameterizeRegionWorkspaceMem(rrw);

    assert(beliefs.size() == nGraphVars * graphVarCardinality);

    varState.resize(nGraphVars, -1);
    varBeliefs.resize(nGraphVars * graphVarCardinality, T(-1));
    maxBelief.resize(nGraphVars, T(-1));

    for(int k = 0; k < nGraphVars * graphVarCardinality; ++k) {
        T val = beliefs[k];
        varBeliefs[k] = val;
        int varIX = k / graphVarCardinality;
        if(val > maxBelief[varIX]) {
            maxBelief[varIX] = val;
            varState[varIX] = k % graphVarCardinality;
        }
    }

    if (flagSaveFullBeliefs) {
        std::ofstream outFileStream(fileAllBeliefs, std::ios_base::binary | std::ios_base::app);
        if (!outFileStream.is_open()) {
            std::cout << "Unable to open binary file for saving all beliefs." << std::endl;
            exit(EXIT_FAILURE);
        }
        outFileStream.write((const char*)&beliefs[0], beliefs.size() * sizeof(float));
        outFileStream.close();

    }

    return true;
}

template <typename T, typename S>
bool MPSolver<T, S>::CheckGraph(const int& nExpectedRegions, const int& nExpextedLamdaSize, const int& globalDebugFlag, const std::string& prompt)
{
    if (globalDebugFlag == 1) {
        std::cout << prompt + "[MP Check] #regions: expexcted " << nExpectedRegions << " while actual " << graph->GetGraphSize() << std::endl;
        std::cout << prompt + "[MP Check] #lambda: expexcted " << nExpextedLamdaSize << " while actual " << graph->GetLambdaSize() << std::endl;
    }
    bool flagNRegions = nExpectedRegions == graph->GetGraphSize();
    bool flagLambdaSize = nExpextedLamdaSize == graph->GetLambdaSize();
    return flagNRegions && flagLambdaSize;
}


template class MPSolver<double, int>;
template class MPSolver<float, int>;

}