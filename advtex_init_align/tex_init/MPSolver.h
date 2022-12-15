#ifndef H_MPSOLVER
#define H_MPSOLVER

#include <vector>

#include "Region.h"

namespace NS_MPSolver {

template <typename T, typename S>
class MPSolver {
public:
    // Currently, we assume each variable has same number of possible states
    S nGraphVars;
    S graphVarCardinality;

    std::vector<S> varState;
    std::vector<T> varBeliefs;
    std::vector<T> maxBelief;

    enum GraphUpdateTypes {
        OnlyUnary = 0,
        OnlyPair = 1,
        BothUnaryPair = 2
    };
private:
    // we need global potentialIDs and regionIDs for any update to graph
    std::vector<typename MPGraph<T, S>::PotentialID> potentialIDs;
    std::vector<typename MPGraph<T, S>::RegionID> regionIDs;

    std::unique_ptr<MPGraph<T, S>> graph;
    // MPGraph<T, S> graph;
public:
    MPSolver();
    ~MPSolver();

    void DestroyGraph();

    bool CreateGraph(const S& numVars, const S& cardinality);
    bool UpdateGraph(const int& updateType, std::vector<T>& unaryInfos,
                     const std::vector<T>& pairInfos, std::vector<T>& pairPotentials);

    bool CheckGraph(const int& nExpectedRegions, const int& nExpextedLamdaSize, const int& globalDebugFlag, const std::string& prompt);

    bool Solve(const int& nIters, const bool& flagSaveFullBeliefs = false, const std::string& fileAlBeliefs = std::string("none"));
};


} //end namespace

#endif