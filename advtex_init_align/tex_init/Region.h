#ifndef H_REGION
#define H_REGION

#include <iostream>
#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include <thread>
#include <memory>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <iomanip>
#include <functional>
#include <string.h>
#include <assert.h>

#include "CPrecisionTimer.h"
//#include "fastonebigheader.h"

//#include <omp.h>

#define NEW_CODE
//#define WITH_ALTERNATING

template <typename T, typename S>
class MPGraph {
public:
	struct PotentialID {
		S PotID;
	};
	struct RegionID {
		S RegID;
	};
	struct DualWorkspaceID {
		T* DualWorkspace;
	};
	struct RRegionAlternateWorkspaceID {
		T* MuPotStore;
		T* MuMem;
		S* MuIXMem;
		S* IXMem;
	};
	struct RRegionWorkspaceID {
		T* RRegionMem;
		T* MuMem;
		S* MuIXMem;
		S* IXMem;
	};
	struct REdgeWorkspaceID {
		T* MuMem;
		S* MuIXMem;
		S* IXMem;
	};
	struct GEdgeWorkspaceID {
		T* mem1;
		T* mem2;
	};
	struct FunctionUpdateWorkspaceID {
		T* mem;
	};
	struct PotentialVector {
		const T* data;
		S size;
		PotentialVector(T* d, S s) : data(d), size(s) {};
		const PotentialVector& operator=(const PotentialVector& o) {
			this->data = o.data;
			this->size = o.size;
			return *this;
		}
	};
    T entropy;
    T* region_belief_sums;
private:
	class Region {
	public:
		const T c_r;
		const T weight_r;
		T sum_c_r_c_p;
		const PotentialVector* const pot;
		const S potSize;
		void* tmp;
		size_t belief;
		std::vector<S> varIX;
	public:
		Region(T c_r, T weight_r, PotentialVector *pot, S potSize, const std::vector<S>& varIX) : c_r(c_r), weight_r(weight_r), pot(pot), potSize(potSize), tmp(NULL), belief(0), varIX(varIX) {};
		virtual ~Region() {};

		S GetPotentialSize() {
			return potSize;
		}
	};

	struct MPNode;
	struct EdgeID;

	struct MsgContainer {
		size_t lambda;
		struct MPNode* node;
		struct EdgeID* edge;
#ifdef NEW_CODE
		S* Translator;
		MsgContainer(size_t l, MPNode* n, EdgeID* e, S* Trans) : lambda(l), node(n), edge(e), Translator(Trans) {};
#else
		std::vector<S> Translator;
		MsgContainer(size_t l, MPNode* n, EdgeID* e, const std::vector<S>& Trans) : lambda(l), node(n), edge(e), Translator(Trans) {};
#endif
	};

	struct MPNode : public Region {
		MPNode(T c_r, T weight_r, const std::vector<S>& varIX, PotentialVector *pot, S potSize) : Region(c_r, weight_r, pot, potSize, varIX) {};
		std::vector<MsgContainer> Parents;
		std::vector<MsgContainer> Children;
	};

	std::vector<S> Cardinalities;
	std::vector<MPNode*> Graph;
	std::vector<size_t> ValidRegionMapping;
	std::vector<size_t> RegionsWithChildren;

	std::vector<PotentialVector*> Potentials;
	size_t LambdaSize;

	struct EdgeID {
		MsgContainer* parentPtr;
		MsgContainer* childPtr;
		std::vector<S> rStateMultipliers;//cumulative size of parent region variables that overlap with child region (r)
		std::vector<S> newVarStateMultipliers;//cumulative size of parent region variables that are unique to parent region
		//std::vector<S> newVarCumSize;//cumulative size of new variables
		std::vector<S> newVarIX;//variable indices of new variables
		S newVarSize;
	};
	std::vector<EdgeID*> Edges;

#ifdef NEW_CODE
	std::vector<S*> ChildTranslators, ParentTranslators;
	std::vector<size_t> ChildTranslatorsSize, ParentTranslatorsSize;
#endif

public:
	MPGraph() : LambdaSize(0), entropy(0), region_belief_sums(NULL) {};
	virtual ~MPGraph() {
		for (typename std::vector<MPNode*>::iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
			MPNode* r_ptr = *r;
			delete r_ptr;
		}

		for (typename std::vector<EdgeID*>::iterator e = Edges.begin(), e_e = Edges.end(); e != e_e; ++e) {
			EdgeID* e_ptr = *e;
			delete e_ptr;
		}
        if (region_belief_sums != NULL) {
            delete [] region_belief_sums;
        }
		for(typename std::vector<PotentialVector*>::iterator p=Potentials.begin(),p_e=Potentials.end();p!=p_e;++p) {
			delete *p;
		}

#ifdef NEW_CODE
		for(typename std::vector<S*>::iterator iter=std::begin(ChildTranslators);iter!=std::end(ChildTranslators);++iter) {
			delete[] *iter;
		}
		for(typename std::vector<S*>::iterator iter=std::begin(ParentTranslators);iter!=std::end(ParentTranslators);++iter) {
			delete[] *iter;
		}
#endif
	};

	// newly-added function
	int GetGraphSize() const {
		return Graph.size();
	}

	int AddVariables(const std::vector<S>& card) {
		Cardinalities = card;
		return 0;
	};
	const PotentialID AddPotential(const PotentialVector& potVals) {
		S PotID = S(Potentials.size());
		Potentials.push_back(new PotentialVector(potVals));
		return PotentialID{ PotID };
	};
	void ReplacePotential(const PotentialVector& potVals, int ind) {
		*Potentials[ind] = potVals;
	}
	const RegionID AddRegion(T c_r, T weight_r, const std::vector<S>& varIX, const PotentialID& p) {
		S RegID = S(Graph.size());
		Graph.push_back(new MPNode(c_r, weight_r, varIX, Potentials[p.PotID], Potentials[p.PotID]->size));
		return RegionID{ RegID };
	};
	int AddConnection(const RegionID& child, const RegionID& parent) {
		MPNode* c = Graph[child.RegID];
		MPNode* p = Graph[parent.RegID];

		LambdaSize += c->GetPotentialSize();

		//c->Parents.push_back(MsgContainer{ 0, p, NULL, std::vector<S>() });
		//p->Children.push_back(MsgContainer{ 0, c, NULL, std::vector<S>() });
#ifdef NEW_CODE
		c->Parents.push_back(MsgContainer(0, p, NULL, NULL));
		p->Children.push_back(MsgContainer(0, c, NULL, NULL));
#else
		c->Parents.push_back(MsgContainer(0, p, NULL, std::vector<S>()));
		p->Children.push_back(MsgContainer(0, c, NULL, std::vector<S>()));
#endif
		return 0;
	};
	int AllocateMessageMemory() {

		size_t lambdaOffset = 0;

		for (typename std::vector<MPNode*>::iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
			MPNode* r_ptr = *r;
			size_t numEl = r_ptr->GetPotentialSize();
			if (r_ptr->Parents.size() > 0) {
				ValidRegionMapping.push_back(r - Graph.begin());
			}
			if (r_ptr->Children.size()>0) {
				RegionsWithChildren.push_back(r - Graph.begin());
			}

			for (typename std::vector<MsgContainer>::iterator pn = r_ptr->Parents.begin(), pn_e = r_ptr->Parents.end(); pn != pn_e; ++pn) {
				MPNode* pn_ptr = pn->node;

				pn->lambda = lambdaOffset;
				for (typename std::vector<MsgContainer>::iterator cn = pn_ptr->Children.begin(), cn_e = pn_ptr->Children.end(); cn != cn_e; ++cn) {
					if (cn->node == r_ptr) {
						cn->lambda = lambdaOffset;
						Edges.push_back(new EdgeID{ &(*pn), &(*cn), std::vector<S>(), std::vector<S>(), std::vector<S>(), 0 });
						pn->edge = cn->edge = Edges.back();
						break;
					}
				}
				lambdaOffset += numEl;
			}
		}
		FillTranslator();
		for (typename std::vector<MPNode*>::iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
			MPNode* r_ptr = *r;
			T sum_c_p = r_ptr->c_r;
			for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p) {
				sum_c_p += p->node->c_r;
			}
			r_ptr->sum_c_r_c_p = sum_c_p;
		}
		return 0;
	}

	const DualWorkspaceID AllocateDualWorkspaceMem(T epsilon) const {
		size_t maxMem = 0;
		for (typename std::vector<MPNode*>::const_iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
			MPNode* r_ptr = *r;
			T ecr = epsilon*r_ptr->c_r;
			size_t s_r_e = r_ptr->GetPotentialSize();

			if (ecr != T(0)) {
				maxMem = (s_r_e > maxMem) ? s_r_e : maxMem;
			}
		}

		return DualWorkspaceID{ ((maxMem > 0) ? new T[maxMem] : NULL) };
	}
	void DeAllocateDualWorkspaceMem(DualWorkspaceID& dw) const {
		delete[] dw.DualWorkspace;
	}

	T* GetMaxMemComputeMu(T epsilon) const {
		size_t maxMem = 0;
		for (typename std::vector<EdgeID*>::const_iterator e = Edges.begin(), e_e = Edges.end(); e != e_e; ++e) {
			EdgeID* edge = *e;
			size_t s_p_e = edge->newVarSize;
			MPNode* p_ptr = edge->parentPtr->node;

			T ecp = epsilon*p_ptr->c_r;
			if (ecp != T(0)) {
				maxMem = (s_p_e > maxMem) ? s_p_e : maxMem;
			}
		}
		return ((maxMem > 0) ? new T[maxMem] : NULL);
	}
	S* GetMaxMemComputeMuIXVar() const {
		size_t maxMem = 0;
		for (size_t r = 0; r < ValidRegionMapping.size(); ++r) {
			MPNode* r_ptr = Graph[ValidRegionMapping[r]];
			for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p) {
				size_t tmp = p->edge->newVarIX.size();
				maxMem = (tmp>maxMem) ? tmp : maxMem;
			}
		}
		return ((maxMem > 0) ? new S[maxMem] : NULL);
	}

#ifdef WITH_ALTERNATING
	const RRegionAlternateWorkspaceID AllocateReparameterizeRegionAlternateWorkspaceMem(T epsilon) const {
		size_t maxMem = 0;
		size_t maxIXMem = 0;
		for (size_t r = 0; r < Graph.size(); ++r) {
			MPNode* r_ptr = Graph[r];
			size_t psz = r_ptr->GetPotentialSize();
			maxMem = (psz>maxMem) ? psz : maxMem;
			size_t rvIX = r_ptr->varIX.size();
			maxIXMem = (rvIX>maxIXMem) ? rvIX : maxIXMem;
		}

		return RRegionAlternateWorkspaceID{ ((maxMem > 0) ? new T[maxMem] : NULL), GetMaxMemComputeMu(epsilon), GetMaxMemComputeMuIXVar(), ((maxIXMem > 0) ? new S[maxIXMem] : NULL) };
	}
	void DeAllocateReparameterizeRegionAlternateWorkspaceMem(RRegionAlternateWorkspaceID& w) const {
		delete[] w.MuPotStore;
		delete[] w.MuMem;
		delete[] w.MuIXMem;
		delete[] w.IXMem;
		w.MuPotStore = NULL;
		w.MuMem = NULL;
		w.MuIXMem = NULL;
		w.IXMem = NULL;
	}
#endif

	const RRegionWorkspaceID AllocateReparameterizeRegionWorkspaceMem(T epsilon) const {
		size_t maxMem = 0;
		size_t maxIXMem = 0;
		for (size_t r = 0; r < ValidRegionMapping.size(); ++r) {
			MPNode* r_ptr = Graph[ValidRegionMapping[r]];
			size_t psz = r_ptr->Parents.size();
			maxMem = (psz>maxMem) ? psz : maxMem;
			size_t rvIX = r_ptr->varIX.size();
			maxIXMem = (rvIX>maxIXMem) ? rvIX : maxIXMem;
		}

		return RRegionWorkspaceID{ ((maxMem > 0) ? new T[maxMem] : NULL), GetMaxMemComputeMu(epsilon), GetMaxMemComputeMuIXVar(), ((maxIXMem > 0) ? new S[maxIXMem] : NULL) };
	}
	void DeAllocateReparameterizeRegionWorkspaceMem(RRegionWorkspaceID& w) const {
		delete[] w.RRegionMem;
		delete[] w.MuMem;
		delete[] w.MuIXMem;
		delete[] w.IXMem;
		w.RRegionMem = NULL;
		w.MuMem = NULL;
		w.MuIXMem = NULL;
		w.IXMem = NULL;
	}

	const REdgeWorkspaceID AllocateReparameterizeEdgeWorkspaceMem(T epsilon) const {
		size_t maxIXMem = 0;
		for(typename std::vector<EdgeID*>::const_iterator eb=Edges.begin(), eb_e=Edges.end();eb!=eb_e;++eb) {
			MPNode* r_ptr = (*eb)->childPtr->node;
			size_t rNumVar = r_ptr->varIX.size();
			maxIXMem = (rNumVar>maxIXMem) ? rNumVar : maxIXMem;
		}
		return REdgeWorkspaceID{ GetMaxMemComputeMu(epsilon), GetMaxMemComputeMuIXVar(), ((maxIXMem > 0) ? new S[maxIXMem] : NULL) };
	}
	void DeAllocateReparameterizeEdgeWorkspaceMem(REdgeWorkspaceID& w) const {
		delete[] w.MuMem;
		delete[] w.MuIXMem;
		delete[] w.IXMem;
		w.MuMem = NULL;
		w.MuIXMem = NULL;
		w.IXMem = NULL;
	}

	const GEdgeWorkspaceID AllocateGradientEdgeWorkspaceMem() const {
		size_t memSZ = 0;
		for (typename std::vector<MPNode*>::const_iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
			MPNode* r_ptr = *r;
			size_t s_r_e = r_ptr->GetPotentialSize();
			memSZ = (s_r_e > memSZ) ? s_r_e : memSZ;
		}
		return GEdgeWorkspaceID{ ((memSZ>0) ? new T[memSZ] : NULL), ((memSZ>0) ? new T[memSZ] : NULL) };
	}
	void DeAllocateGradientEdgeWorkspaceMem(GEdgeWorkspaceID& w) const {
		delete[] w.mem1;
		delete[] w.mem2;
	}

	const FunctionUpdateWorkspaceID AllocateFunctionUpdateWorkspaceMem() const {
		size_t memSZ = 0;
		for (typename std::vector<MPNode*>::const_iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
			MPNode* r_ptr = *r;
			size_t s_r_e = r_ptr->GetPotentialSize();
			memSZ = (s_r_e > memSZ) ? s_r_e : memSZ;
		}
		return FunctionUpdateWorkspaceID{ ((memSZ>0) ? new T[memSZ] : NULL)};
	}
	void DeAllocateFunctionUpdateWorkspaceID(FunctionUpdateWorkspaceID& w) const {
		delete[] w.mem;
	}

	int FillEdge() {
		for (typename std::vector<EdgeID*>::iterator e = Edges.begin(), e_e = Edges.end(); e != e_e; ++e) {
			MPNode* r_ptr = (*e)->childPtr->node;
			MPNode* p_ptr = (*e)->parentPtr->node;

			(*e)->newVarSize = 1;
			(*e)->rStateMultipliers.assign(r_ptr->varIX.size(), 1);
			for (typename std::vector<S>::iterator vp = p_ptr->varIX.begin(), vp_e = p_ptr->varIX.end(); vp != vp_e; ++vp) {
				typename std::vector<S>::iterator vr = std::find(r_ptr->varIX.begin(), r_ptr->varIX.end(), *vp);
				size_t posP = vp - p_ptr->varIX.begin();
				if (vr == r_ptr->varIX.end()) {
					(*e)->newVarIX.push_back(*vp);
					(*e)->newVarStateMultipliers.push_back(((std::vector<S>*)p_ptr->tmp)->at(posP));
					(*e)->newVarSize *= Cardinalities[*vp];
				} else {
					size_t posR = vr - r_ptr->varIX.begin();
					(*e)->rStateMultipliers[posR] = ((std::vector<S>*)p_ptr->tmp)->at(posP);
				}
			}

			/*e->newVarCumSize.assign(e->newVarIX.size(), 1);
			for (size_t k = 1, k_e = e->newVarIX.size(); k < k_e; ++k) {
				e->newVarCumSize[k] = e->newVarCumSize[k - 1] * Cardinalities[e->newVarIX[k]];
			}*/
		}
		return 0;
	}

	void ComputeCumulativeSize(MPNode* r_ptr, std::vector<S>& cumVarR) {
		size_t numVars = r_ptr->varIX.size();
		cumVarR.assign(numVars, 1);
		for (size_t v = 1; v < numVars; ++v) {
			cumVarR[v] = cumVarR[v - 1] * Cardinalities[r_ptr->varIX[v-1]];
		}
	}

	int FillTranslator() {
		for (typename std::vector<MPNode*>::iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
			MPNode* r_ptr = *r;
			std::vector<S>* cumVarR = new std::vector<S>;
			ComputeCumulativeSize(r_ptr, *cumVarR);
			r_ptr->tmp = (void*)cumVarR;
		}
		FillEdge();

#ifdef NEW_CODE		
		struct TranslatorIdentifier {
			std::vector<S>* parentCumVar;
			std::vector<S>* childCumVar;
			std::vector<S>* childVarPosInParent;
			S parentPotSize;
			S childPotSize;
			bool operator ==(const TranslatorIdentifier &b) const {
				return (*(this->parentCumVar) == *(b.parentCumVar)) && (*(this->childCumVar) == *(b.childCumVar)) && (*(this->childVarPosInParent) == *(b.childVarPosInParent)) && this->parentPotSize==b.parentPotSize && this->childPotSize==b.childPotSize;
			}
		};
		std::vector<TranslatorIdentifier> TIList;

		for (typename std::vector<MPNode*>::iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
			MPNode* r_ptr = *r;

			for (typename std::vector<MsgContainer>::iterator cn = r_ptr->Children.begin(), cn_e = r_ptr->Children.end(); cn != cn_e; ++cn) {
				MPNode* c_ptr = cn->node;
				size_t numVarsC = c_ptr->varIX.size();

				std::vector<S>* childVarPosInParent = new std::vector<S>(numVarsC);

				for (size_t cIX = 0; cIX < numVarsC; ++cIX) {
					S curVar = c_ptr->varIX[cIX];
					typename std::vector<S>::iterator iter = std::find(r_ptr->varIX.begin(), r_ptr->varIX.end(), curVar);
					assert(iter != r_ptr->varIX.end());
					S pos = iter - r_ptr->varIX.begin();
					(*childVarPosInParent)[cIX] = pos;
				}

				TranslatorIdentifier ti{(std::vector<S>*)r_ptr->tmp, (std::vector<S>*)c_ptr->tmp, childVarPosInParent, r_ptr->GetPotentialSize(), c_ptr->GetPotentialSize()};

				size_t loopIX=0;
				for (;loopIX<TIList.size();++loopIX) {
					if(TIList[loopIX]==ti) {
						break;
					}
				}
				if(loopIX==TIList.size()) {
					//create new translator
					TIList.push_back(ti);
					size_t numEl = ti.parentPotSize;
					S* translator = new S[numEl];
					for (size_t s_r = 0; s_r < numEl; ++s_r) {
						S val = 0;
						for (size_t cIX = 0; cIX < ti.childCumVar->size(); ++cIX) {
							size_t pos = ti.childVarPosInParent->at(cIX);
							S curState = (s_r / (ti.parentCumVar->at(pos))) % Cardinalities[r_ptr->varIX[pos]];
							val += curState*ti.childCumVar->at(cIX);
						}
						translator[s_r] = val;
					}
					ChildTranslators.push_back(translator);
					ChildTranslatorsSize.push_back(numEl);
					cn->Translator = translator;
				} else {
					//assign old translator
					delete childVarPosInParent;
					cn->Translator = ChildTranslators[loopIX];
				}
			}
		}
		for(size_t k=0;k<TIList.size();++k) {
			delete TIList[k].childVarPosInParent;
		}
		std::vector<TranslatorIdentifier>().swap(TIList);

		for (typename std::vector<MPNode*>::iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
			MPNode* r_ptr = *r;

			for (typename std::vector<MsgContainer>::iterator pn = r_ptr->Parents.begin(), pn_e = r_ptr->Parents.end(); pn != pn_e; ++pn) {
                MPNode* c_ptr = r_ptr;
				MPNode* p_ptr = pn->node;
				size_t numVarsC = c_ptr->varIX.size();

				std::vector<S>* childVarPosInParent = new std::vector<S>(numVarsC);

				for (size_t cIX = 0; cIX < numVarsC; ++cIX) {
					S curVar = c_ptr->varIX[cIX];
					typename std::vector<S>::iterator iter = std::find(p_ptr->varIX.begin(), p_ptr->varIX.end(), curVar);
					size_t pos = iter - p_ptr->varIX.begin();
					assert(iter != p_ptr->varIX.end());
					(*childVarPosInParent)[cIX] = pos;
				}

				TranslatorIdentifier ti{(std::vector<S>*)p_ptr->tmp, (std::vector<S>*)c_ptr->tmp, childVarPosInParent, p_ptr->GetPotentialSize(), c_ptr->GetPotentialSize() };

				size_t loopIX=0;
				for (;loopIX<TIList.size();++loopIX) {
					if(TIList[loopIX]==ti) {
						break;
					}
				}
				if(loopIX==TIList.size()) {
					TIList.push_back(ti);
					size_t numEl = ti.parentPotSize;
					S* translator = new S[numEl];
					for (size_t s_r = 0; s_r < numEl; ++s_r) {
						S val = 0;
						for (size_t cIX = 0; cIX < ti.childCumVar->size(); ++cIX) {
							size_t pos = ti.childVarPosInParent->at(cIX);
							S curState = (s_r / (ti.parentCumVar->at(pos))) % Cardinalities[p_ptr->varIX[pos]];
							val += curState*ti.childCumVar->at(cIX);
						}
						translator[s_r] = val;
					}
					ParentTranslators.push_back(translator);
					ParentTranslatorsSize.push_back(numEl);
					pn->Translator = translator;
				} else {
					delete childVarPosInParent;
					pn->Translator = ParentTranslators[loopIX];
				}
			}
		}
		for(size_t k=0;k<TIList.size();++k) {
			delete TIList[k].childVarPosInParent;
		}
		std::vector<TranslatorIdentifier>().swap(TIList);
#else

		for (typename std::vector<MPNode*>::iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
			MPNode* r_ptr = *r;
			size_t numEl = r_ptr->GetPotentialSize();

			for (typename std::vector<MsgContainer>::iterator cn = r_ptr->Children.begin(), cn_e = r_ptr->Children.end(); cn != cn_e; ++cn) {
				MPNode* c_ptr = cn->node;
				size_t numVarsC = c_ptr->varIX.size();

				cn->Translator.assign(numEl, S(0));

				for (size_t s_r = 0; s_r < numEl; ++s_r) {
					S val = 0;
					for (size_t cIX = 0; cIX < numVarsC; ++cIX) {
						S curVar = c_ptr->varIX[cIX];
						typename std::vector<S>::iterator iter = std::find(r_ptr->varIX.begin(), r_ptr->varIX.end(), curVar);
						size_t pos = iter - r_ptr->varIX.begin();
						assert(iter != r_ptr->varIX.end());
						S curState = (s_r / (((std::vector<S>*)r_ptr->tmp)->at(pos))) % Cardinalities[r_ptr->varIX[pos]];
						val += curState*((std::vector<S>*)c_ptr->tmp)->at(cIX);
					}

					cn->Translator[s_r] = val;
				}
			}
            for (typename std::vector<MsgContainer>::iterator pn = r_ptr->Parents.begin(), pn_e = r_ptr->Parents.end(); pn != pn_e; ++pn) {
                MPNode* c_ptr = r_ptr;
				MPNode* p_ptr = pn->node;
				size_t numVarsC = c_ptr->varIX.size();

			    numEl = p_ptr->GetPotentialSize();
				pn->Translator.assign(numEl, S(0));

				for (size_t s_p = 0; s_p < numEl; ++s_p) {
					S val = 0;
					for (size_t cIX = 0; cIX < numVarsC; ++cIX) {
						S curVar = c_ptr->varIX[cIX];
						typename std::vector<S>::iterator iter = std::find(p_ptr->varIX.begin(), p_ptr->varIX.end(), curVar);
						size_t pos = iter - p_ptr->varIX.begin();
						assert(iter != p_ptr->varIX.end());
						S curState = (s_p / (((std::vector<S>*)p_ptr->tmp)->at(pos))) % Cardinalities[p_ptr->varIX[pos]];
						val += curState*((std::vector<S>*)c_ptr->tmp)->at(cIX);
					}

					pn->Translator[s_p] = val;
				}
			}

		}
#endif
		for (typename std::vector<MPNode*>::iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
			MPNode* r_ptr = *r;
			std::vector<S>* cumVarR = (std::vector<S>*)r_ptr->tmp;
			delete cumVarR;
			r_ptr->tmp = NULL;
		}
		return 0;
	}

	size_t NumberOfRegionsTotal() const {
		return Graph.size();
	}

	size_t NumberOfRegionsWithParents() const {
		return ValidRegionMapping.size();
	}

	size_t NumberOfRegionsWithChildren() const {
		return RegionsWithChildren.size();
	}

	size_t NumberOfEdges() const {
		return Edges.size();
	}

	void UpdateEdge(T* lambdaBase, T* lambdaGlobal, int e, bool additiveUpdate) {
		if (lambdaBase == lambdaGlobal) {
			assert(additiveUpdate == false);//change ReparameterizationEdge function to directly perform update and don't call UpdateEdge
			return;
		}

		EdgeID* edge = Edges[e];
		MPNode* r_ptr = edge->childPtr->node;
		size_t s_r_e = r_ptr->GetPotentialSize();

		for (size_t s_r = 0; s_r < s_r_e; ++s_r) {
			if (additiveUpdate) {
				lambdaGlobal[edge->childPtr->lambda + s_r] += lambdaBase[edge->childPtr->lambda + s_r];
			} else {
				lambdaGlobal[edge->childPtr->lambda + s_r] = lambdaBase[edge->childPtr->lambda + s_r];
			}
		}
	}

	void CopyLambda(T* lambdaSrc, T* lambdaDst, size_t s_r_e) const {
		//std::copy(lambdaSrc, lambdaSrc + s_r_e, lambdaDst);
		//memcpy((void*)(lambdaDst), (void*)(lambdaSrc), s_r_e*sizeof(T));
		for (T* ptr_e = lambdaSrc + s_r_e; ptr_e != lambdaSrc;) {
			*lambdaDst++ = *lambdaSrc++;
		}
	}

	void CopyMessagesForLocalFunction(T* lambdaSrc, T* lambdaDst, int r) const {
		MPNode* r_ptr = Graph[r];
		size_t s_r_e = r_ptr->GetPotentialSize();

		for (typename std::vector<MsgContainer>::const_iterator pn = r_ptr->Parents.begin(), pn_e = r_ptr->Parents.end(); pn != pn_e; ++pn) {
			CopyLambda(lambdaSrc + pn->lambda, lambdaDst + pn->lambda, s_r_e);
		}

		for (typename std::vector<MsgContainer>::const_iterator cn = r_ptr->Children.begin(), cn_e = r_ptr->Children.end(); cn != cn_e; ++cn) {
			size_t s_r_c = cn->node->GetPotentialSize();
			CopyLambda(lambdaSrc + cn->lambda, lambdaDst + cn->lambda, s_r_c);
		}
	}

	void ComputeLocalFunctionUpdate(T* lambdaBase, int r, T epsilon, T multiplier, bool additiveUpdate, FunctionUpdateWorkspaceID& w) {
		assert(additiveUpdate==true);
		MPNode* r_ptr = Graph[r];
		size_t s_r_e = r_ptr->GetPotentialSize();

		T c_r = r_ptr->c_r;
		T frac = multiplier;
		if(epsilon>0) {
				frac /= (epsilon*c_r);
		}

		T* mem = w.mem;
		//ComputeBeliefForRegion(r_ptr, lambdaBase, epsilon, mem, s_r_e);
		for(size_t s_r=0;s_r<s_r_e;++s_r) {
			mem[s_r] *= frac;
		}

		for (typename std::vector<MsgContainer>::iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p) {
			for (size_t s_r = 0; s_r < s_r_e; ++s_r) {
				lambdaBase[p->lambda+s_r] = -mem[s_r];
			}
		}

		for (typename std::vector<MsgContainer>::const_iterator c = r_ptr->Children.begin(), c_e = r_ptr->Children.end(); c != c_e; ++c) {
            size_t s_r_c = c->node->GetPotentialSize();
            std::fill_n(lambdaBase+c->lambda, s_r_c, T(0));
			for(size_t s_r = 0;s_r < s_r_e;++s_r) {
				lambdaBase[c->lambda+c->Translator[s_r]] += mem[s_r];
			}
		}
	}

	void UpdateLocalFunction(T* lambdaBase, T* lambdaGlobal, int r, bool additiveUpdate) {
		if (lambdaBase == lambdaGlobal) {
			return;
		}
		MPNode* r_ptr = Graph[r];
		size_t s_r_e = r_ptr->GetPotentialSize();

		if (additiveUpdate) {
			for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p) {
				for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
					lambdaGlobal[p->lambda + s_r] += lambdaBase[p->lambda + s_r];
				}
			}

			for (typename std::vector<MsgContainer>::const_iterator cn = r_ptr->Children.begin(), cn_e = r_ptr->Children.end(); cn != cn_e; ++cn) {
				size_t s_r_c = cn->node->GetPotentialSize();
				for(size_t s_c = 0; s_c != s_r_c; ++s_c) {
					lambdaGlobal[cn->lambda + s_c] += lambdaBase[cn->lambda + s_c];
				}
			}
		} else {
			assert(false);
		}
	}

	void CopyMessagesForEdge(T* lambdaSrc, T* lambdaDst, int e) const {
		EdgeID* edge = Edges[e];
		MPNode* r_ptr = edge->childPtr->node;
		MPNode* p_ptr = edge->parentPtr->node;

		size_t s_r_e = r_ptr->GetPotentialSize();
		size_t s_p_e = p_ptr->GetPotentialSize();

		for (typename std::vector<MsgContainer>::const_iterator pn = r_ptr->Parents.begin(), pn_e = r_ptr->Parents.end(); pn != pn_e; ++pn) {
			CopyLambda(lambdaSrc + pn->lambda, lambdaDst + pn->lambda, s_r_e);
		}

		for (typename std::vector<MsgContainer>::const_iterator cn = r_ptr->Children.begin(), cn_e = r_ptr->Children.end(); cn != cn_e; ++cn) {
			size_t s_r_c = cn->node->GetPotentialSize();
			CopyLambda(lambdaSrc + cn->lambda, lambdaDst + cn->lambda, s_r_c);
		}

		for (typename std::vector<MsgContainer>::const_iterator p_hat = p_ptr->Parents.begin(), p_hat_e = p_ptr->Parents.end(); p_hat != p_hat_e; ++p_hat) {
			CopyLambda(lambdaSrc + p_hat->lambda, lambdaDst + p_hat->lambda, s_p_e);
		}

		for (typename std::vector<MsgContainer>::const_iterator c_hat = p_ptr->Children.begin(), c_hat_e = p_ptr->Children.end(); c_hat != c_hat_e; ++c_hat) {
			if (c_hat->node != r_ptr) {
				size_t s_r_pc = c_hat->node->GetPotentialSize();
				CopyLambda(lambdaSrc + c_hat->lambda, lambdaDst + c_hat->lambda, s_r_pc);
			}
		}
	}

	void CopyMessagesForStar(T* lambdaSrc, T* lambdaDst, int r) const {
		MPNode* r_ptr = Graph[ValidRegionMapping[r]];
		size_t s_r_e = r_ptr->GetPotentialSize();

		for (typename std::vector<MsgContainer>::const_iterator pn = r_ptr->Parents.begin(), pn_e = r_ptr->Parents.end(); pn != pn_e; ++pn) {
			CopyLambda(lambdaSrc + pn->lambda, lambdaDst + pn->lambda, s_r_e);

			MPNode* p_ptr = pn->node;
			size_t s_p_e = p_ptr->GetPotentialSize();
			for (typename std::vector<MsgContainer>::const_iterator p_hat = p_ptr->Parents.begin(), p_hat_e = p_ptr->Parents.end(); p_hat != p_hat_e; ++p_hat) {
				CopyLambda(lambdaSrc + p_hat->lambda, lambdaDst + p_hat->lambda, s_p_e);
			}

			for (typename std::vector<MsgContainer>::const_iterator c_hat = p_ptr->Children.begin(), c_hat_e = p_ptr->Children.end(); c_hat != c_hat_e; ++c_hat) {
				if (c_hat->node != r_ptr) {
					size_t s_pc_e = c_hat->node->GetPotentialSize();
					CopyLambda(lambdaSrc + c_hat->lambda, lambdaDst + c_hat->lambda, s_pc_e);
				}
			}
		}

		for (typename std::vector<MsgContainer>::const_iterator cn = r_ptr->Children.begin(), cn_e = r_ptr->Children.end(); cn != cn_e; ++cn) {
			size_t s_r_c = cn->node->GetPotentialSize();
			CopyLambda(lambdaSrc + cn->lambda, lambdaDst + cn->lambda, s_r_c);
		}
	}

	void ReparameterizeEdge(T* lambdaBase, int e, T epsilon, bool additiveUpdate, REdgeWorkspaceID& wspace) {
		EdgeID* edge = Edges[e];
		MPNode* r_ptr = edge->childPtr->node;
		MPNode* p_ptr = edge->parentPtr->node;

		size_t s_r_e = r_ptr->GetPotentialSize();

		T c_p = p_ptr->c_r;
		T c_r = r_ptr->c_r;
		T frac = T(1) / (c_p + c_r);

		size_t rNumVar = r_ptr->varIX.size();
		//std::vector<S> indivVarStates(rNumVar, 0);
		S* indivVarStates = wspace.IXMem;
		for(S *tmp=indivVarStates, *tmp_e=indivVarStates+rNumVar;tmp!=tmp_e;++tmp) {
			*tmp = 0;
		}
		for (size_t s_r = 0; s_r < s_r_e; ++s_r) {
			if (additiveUpdate) {//additive update
				T updateVal1 = ComputeReparameterizationPotential(lambdaBase, r_ptr, s_r);
				T updateVal2 = lambdaBase[edge->childPtr->lambda + s_r] + ComputeMu(lambdaBase, edge, indivVarStates, rNumVar, epsilon, wspace.MuMem, wspace.MuIXMem);
				//lambdaBase[edge->childPtr->lambda+s_r] += (c_p*frac*updateVal1 - c_r*frac*updateVal2);//use this line if global memory is equal to local memory
				lambdaBase[edge->childPtr->lambda + s_r] = (c_p*frac*updateVal1 - c_r*frac*updateVal2);//addition will be performed in UpdateEdge function
			} else {//absolute update
				T updateVal1 = ComputeReparameterizationPotential(lambdaBase, r_ptr, s_r) + lambdaBase[edge->childPtr->lambda+s_r];
				T updateVal2 = ComputeMu(lambdaBase, edge, indivVarStates, rNumVar, epsilon, wspace.MuMem, wspace.MuIXMem);
				lambdaBase[edge->childPtr->lambda + s_r] = (c_p*frac*updateVal1 - c_r*frac*updateVal2);
			}

			for (size_t varIX = 0; varIX < rNumVar; ++varIX) {
				++indivVarStates[varIX];
				if (indivVarStates[varIX] == Cardinalities[r_ptr->varIX[varIX]]) {
					indivVarStates[varIX] = 0;
				} else {
					break;
				}
			}
		}
	}

	//T ComputeMu(T* lambdaBase, EdgeID* edge, const std::vector<S>& indivVarStates, T epsilon, T* workspaceMem, S* MuIXMem) {
	T ComputeMu(T* lambdaBase, EdgeID* edge, S* indivVarStates, size_t numVarsOverlap, T epsilon, T* workspaceMem, S* MuIXMem) {
		MPNode* r_ptr = edge->childPtr->node;
		MPNode* p_ptr = edge->parentPtr->node;

		//size_t numVarsOverlap = indivVarStates.size();
		size_t s_p_stat = 0;
		for (size_t k = 0; k<numVarsOverlap; ++k) {
			s_p_stat += indivVarStates[k]*edge->rStateMultipliers[k];
		}

		//size_t s_p_e = edge->newVarCumSize.back();
		size_t s_p_e = edge->newVarSize;
		size_t numVarNew = edge->newVarIX.size();

		T maxval = -std::numeric_limits<T>::max();
		T ecp = epsilon*p_ptr->c_r;
		/*T* mem = NULL;
		if (ecp != T(0)) {
			mem = new T[s_p_e];
		}*/
		T* mem = workspaceMem;

		//individual vars;
		//std::vector<S> indivNewVarStates(numVarNew, 0);
		for(S *tmpmem=MuIXMem, *tmpmem_e=MuIXMem+numVarNew;tmpmem!=tmpmem_e;++tmpmem) {
			*tmpmem = 0;
		}
		S* indivNewVarStates = MuIXMem;
		size_t s_p_real = s_p_stat;
		for (size_t s_p = 0; s_p != s_p_e; ++s_p) {
			//size_t s_p_real = s_p_stat;
			//for (size_t varIX = 0; varIX<numVarNew; ++varIX) {
			//	s_p_real += indivNewVarStates[varIX]*edge->newVarStateMultipliers[varIX];
			//}

			T buf = (p_ptr->pot->data == NULL) ? T(0) : p_ptr->weight_r*p_ptr->pot->data[s_p_real];

			for (typename std::vector<MsgContainer>::const_iterator p_hat = p_ptr->Parents.begin(), p_hat_e = p_ptr->Parents.end(); p_hat != p_hat_e; ++p_hat) {
				buf -= lambdaBase[p_hat->lambda+s_p_real];
			}

			for (typename std::vector<MsgContainer>::const_iterator c_hat = p_ptr->Children.begin(), c_hat_e = p_ptr->Children.end(); c_hat != c_hat_e; ++c_hat) {
				if (c_hat->node != r_ptr) {
					buf += lambdaBase[c_hat->lambda+c_hat->Translator[s_p_real]];
				}
			}

			if (ecp != T(0)) {
				buf /= ecp;
				mem[s_p] = buf;
			}
			maxval = (buf>maxval) ? buf : maxval;

			for (size_t varIX = 0; varIX < numVarNew; ++varIX) {
				++indivNewVarStates[varIX];
				if (indivNewVarStates[varIX] == Cardinalities[edge->newVarIX[varIX]]) {
					indivNewVarStates[varIX] = 0;
					s_p_real -= (Cardinalities[edge->newVarIX[varIX]]-1)*edge->newVarStateMultipliers[varIX];
				} else {
					s_p_real += edge->newVarStateMultipliers[varIX];
					break;
				}
			}
		}

		if (ecp != T(0)) {
			T sumVal = std::exp(mem[0] - maxval);
			for (size_t s_p = 1; s_p != s_p_e; ++s_p) {
				sumVal += std::exp(mem[s_p] - maxval);
			}
			maxval = ecp*(maxval + std::log(sumVal));
			//delete[] mem;
		}

		return maxval;
	}

	void UpdateRegion(T* lambdaBase, T* lambdaGlobal, int r, bool additiveUpdate) {
		if (lambdaBase == lambdaGlobal) {
			return;
		}
		MPNode* r_ptr = Graph[ValidRegionMapping[r]];

		size_t s_r_e = r_ptr->GetPotentialSize();
		if (additiveUpdate) {
			for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
				for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p) {
					lambdaGlobal[p->lambda + s_r] += lambdaBase[p->lambda + s_r];
				}
			}
		} else {
			for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
				for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p) {
					lambdaGlobal[p->lambda + s_r] = lambdaBase[p->lambda + s_r];
				}
			}
		}
	}

#ifdef WITH_ALTERNATING
	void ReparameterizeRegionAlternateMu(T* lambdaBase, T* muBase, T epsilon, RRegionAlternateWorkspaceID& wspace) {
		T* mem = wspace.MuMem;
		T* muPotStore = wspace.MuPotStore;
		for(size_t r=0;r<Graph.size();++r) {
			MPNode* p_ptr = Graph[r];
			if(p_ptr->Children.size()>0) {
				T ecp = epsilon*p_ptr->c_r;
				size_t s_p_e = p_ptr->GetPotentialSize();
				for (size_t s_p = 0; s_p != s_p_e; ++s_p) {
					T buf = (p_ptr->pot->data == NULL) ? T(0) : p_ptr->weight_r*p_ptr->pot->data[s_p];

					for (typename std::vector<MsgContainer>::const_iterator p_hat = p_ptr->Parents.begin(), p_hat_e = p_ptr->Parents.end(); p_hat != p_hat_e; ++p_hat) {
						buf -= lambdaBase[p_hat->lambda+s_p];
					}

					for (typename std::vector<MsgContainer>::const_iterator c_hat = p_ptr->Children.begin(), c_hat_e = p_ptr->Children.end(); c_hat != c_hat_e; ++c_hat) {
						buf += lambdaBase[c_hat->lambda+c_hat->Translator[s_p]];
					}
					muPotStore[s_p] = buf;
				}

				for (typename std::vector<MsgContainer>::const_iterator c_hat = p_ptr->Children.begin(), c_hat_e = p_ptr->Children.end(); c_hat != c_hat_e; ++c_hat) {
					MPNode* r_ptr = c_hat->node;
					EdgeID* edge = c_hat->edge;

					size_t s_r_e = r_ptr->GetPotentialSize();
					size_t rNumVar = r_ptr->varIX.size();
					S* indivVarStates = wspace.IXMem;
					for(S *tmp=indivVarStates, *tmp_e=indivVarStates+rNumVar;tmp!=tmp_e;++tmp) {
						*tmp = 0;
					}

					for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
						size_t s_p_stat = 0;
						for (size_t k = 0; k<rNumVar; ++k) {
							s_p_stat += indivVarStates[k]*edge->rStateMultipliers[k];
						}
						size_t s_p_e = edge->newVarSize;
						size_t numVarNew = edge->newVarIX.size();

						T maxval = -std::numeric_limits<T>::max();
						for(S *tmpmem=wspace.MuIXMem, *tmpmem_e=wspace.MuIXMem+numVarNew;tmpmem!=tmpmem_e;++tmpmem) {
							*tmpmem = 0;
						}
						S* indivNewVarStates = wspace.MuIXMem;
						size_t s_p_real = s_p_stat;
						for (size_t s_p = 0; s_p != s_p_e; ++s_p) {
							T buf = muPotStore[s_p_real];

							buf -= lambdaBase[c_hat->lambda+c_hat->Translator[s_p_real]];

							if (ecp != T(0)) {
								buf /= ecp;
								mem[s_p] = buf;
							}
							maxval = (buf>maxval) ? buf : maxval;

							for (size_t varIX = 0; varIX < numVarNew; ++varIX) {
								++indivNewVarStates[varIX];
								if (indivNewVarStates[varIX] == Cardinalities[edge->newVarIX[varIX]]) {
									indivNewVarStates[varIX] = 0;
									s_p_real -= (Cardinalities[edge->newVarIX[varIX]]-1)*edge->newVarStateMultipliers[varIX];
								} else {
									s_p_real += edge->newVarStateMultipliers[varIX];
									break;
								}
							}
						}

						if (ecp != T(0)) {
							T sumVal = std::exp(mem[0] - maxval);
							for (size_t s_p = 1; s_p != s_p_e; ++s_p) {
								sumVal += std::exp(mem[s_p] - maxval);
							}
							maxval = ecp*(maxval + std::log(sumVal));
							//delete[] mem;
						}
						
						muBase[edge->childPtr->lambda+s_r] = maxval;

						for (size_t varIX = 0; varIX < rNumVar; ++varIX) {
							++indivVarStates[varIX];
							if (indivVarStates[varIX] == Cardinalities[r_ptr->varIX[varIX]]) {
								indivVarStates[varIX] = 0;
							} else {
								break;
							}
						}
					}
				}
			}
		}
	}

	void ReparameterizeRegionAlternateLambda_SerializeData(S*& translators, std::vector<T>& serializedDataFloat, std::vector<int>& serializedDataInt, std::vector<T>& potentialData, std::vector<int>& DataFloatOffsets, std::vector<int>& DataIntOffsets) {
		size_t transSize = std::accumulate(ChildTranslatorsSize.begin(), ChildTranslatorsSize.end(), 0);
		transSize = std::accumulate(ParentTranslatorsSize.begin(), ParentTranslatorsSize.end(), transSize);
		translators = new S[transSize];
		S* ptr = translators;
		std::map<S*,size_t> OldTranslator2NewTranslator;
		for(size_t k=0;k<ChildTranslators.size();++k) {
			std::copy(ChildTranslators[k], ChildTranslators[k]+ChildTranslatorsSize[k], ptr);
			OldTranslator2NewTranslator.insert(std::pair<S*,size_t>(ChildTranslators[k],size_t(ptr-translators)));
			ptr += ChildTranslatorsSize[k];
		}
		for(size_t k=0;k<ParentTranslators.size();++k) {
			std::copy(ParentTranslators[k], ParentTranslators[k]+ParentTranslatorsSize[k], ptr);
			OldTranslator2NewTranslator.insert(std::pair<S*,size_t>(ParentTranslators[k],size_t(ptr-translators)));
			ptr += ParentTranslatorsSize[k];
		}

		//std::vector<T> serializedDataFloat;
		//std::vector<int> serializedDataInt;
		//std::vector<T> potentialData;
		//std::vector<int> DataFloatOffsets;
		//std::vector<int> DataIntOffsets;
		for (int k = 0, k_e = int(NumberOfRegionsWithParents()); k < k_e; ++k) {
			DataFloatOffsets.push_back(serializedDataFloat.size());
			DataIntOffsets.push_back(serializedDataInt.size());
			MPNode* r_ptr = Graph[ValidRegionMapping[k]];
			std::vector<T> localDataFloat;
			std::vector<int> localDataInt;
			T inv_sum_c_p = T(1.)/r_ptr->sum_c_r_c_p;
			localDataFloat.push_back(inv_sum_c_p);
			size_t s_r_e = r_ptr->GetPotentialSize();
			localDataInt.push_back(s_r_e);
			T weight_r = r_ptr->weight_r;
			localDataFloat.push_back(weight_r);
			T* pot = r_ptr->pot->data; //store entire potential of region;
			assert(pot!=NULL);
			size_t potOffset = potentialData.size();
			localDataInt.push_back(potOffset);
			potentialData.insert(potentialData.end(), pot, pot+s_r_e);

			size_t sz1 = r_ptr->Children.size();
			localDataInt.push_back(sz1);
			size_t sz2 = r_ptr->Parents.size();
			localDataInt.push_back(sz2);
			for (typename std::vector<MsgContainer>::const_iterator c = r_ptr->Children.begin(), c_e = r_ptr->Children.end(); c != c_e; ++c) {
				size_t l = c->lambda;
				localDataInt.push_back(l);
				size_t t = OldTranslator2NewTranslator.find(c->Translator)->second;
				localDataInt.push_back(t);
			}
			
			for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p) {
				size_t m = p->lambda;
				localDataInt.push_back(m);
				T val = p->node->c_r;
				localDataFloat.push_back(val);
			}
			serializedDataFloat.insert(serializedDataFloat.end(), localDataFloat.begin(), localDataFloat.end());
			serializedDataInt.insert(serializedDataInt.end(), localDataInt.begin(), localDataInt.end());
		}

		//push translators, serializedDataFloat, serializedDataInt, potentialData, DataFloatOffsets, DataIntOffsets to GPU
	}

	void ReparameterizeRegionAlternateLambda_Serialized(T* lambdaBase, T* muBase, S* translators, T* serializedDataFloat, S* serializedDataInt, T* potentialData, S* DataFloatOffsets, S* DataIntOffsets, int k_e) {
		for(int k=0;k<k_e;++k) {
			size_t IntStart = DataIntOffsets[k];
			size_t FloatStart = DataFloatOffsets[k];
			T inv_sum_c_p = serializedDataFloat[FloatStart];
			T weight_r = serializedDataFloat[FloatStart+1];
			int s_r_e = serializedDataInt[IntStart];
			int potOffset = serializedDataInt[IntStart+1];
			int numChildren = serializedDataInt[IntStart+2];
			int numParents = serializedDataInt[IntStart+3];
			for(int s_r=0;s_r<s_r_e;++s_r) {
				T phi_r_x_r_prime = weight_r*potentialData[potOffset+s_r];
				for(int n=0;n<numChildren;++n) {
					int lOffset = serializedDataInt[IntStart+4+2*n];
					int tOffset = serializedDataInt[IntStart+5+2*n];
					phi_r_x_r_prime += lambdaBase[lOffset+translators[tOffset+s_r]];
				}
				for(int n=0;n<numParents;++n) {
					int lOffset = serializedDataInt[IntStart+4+2*numChildren+n];
					phi_r_x_r_prime += muBase[lOffset+s_r];
				}
				phi_r_x_r_prime *= inv_sum_c_p;
				for(int n=0;n<numParents;++n) {
					int lOffset = serializedDataInt[IntStart+4+2*numChildren+n];
					T c_p = serializedDataFloat[FloatStart+2+n];
					lambdaBase[lOffset+s_r] = c_p*phi_r_x_r_prime - muBase[lOffset+s_r];
				}
			}
			for(int n=0;n<numParents;++n) {
				int lOffset = serializedDataInt[IntStart+4+2*numChildren+n];
				for(int s_r=s_r_e=1;s_r!=0;--s_r) {
					lambdaBase[lOffset+s_r] -= lambdaBase[lOffset];
				}
				lambdaBase[lOffset] = 0;
			}
		}
	}

	void ReparameterizeRegionAlternateLambda(T* lambdaBase, T* muBase) {
		for (int k = 0, k_e = int(NumberOfRegionsWithParents()); k < k_e; ++k) {
			MPNode* r_ptr = Graph[ValidRegionMapping[k]];
			T inv_sum_c_p = T(1.)/r_ptr->sum_c_r_c_p;
			size_t s_r_e = r_ptr->GetPotentialSize();
			for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
				T phi_r_x_r_prime = (r_ptr->pot->data == NULL) ? T(0) : r_ptr->weight_r*r_ptr->pot->data[s_r];

				for (typename std::vector<MsgContainer>::const_iterator c = r_ptr->Children.begin(), c_e = r_ptr->Children.end(); c != c_e; ++c) {
					phi_r_x_r_prime += lambdaBase[c->lambda+c->Translator[s_r]];
				}

				for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p) {
					phi_r_x_r_prime += muBase[p->lambda+s_r];
				}

				phi_r_x_r_prime *= inv_sum_c_p;

				for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p) {
					MPNode* ptr = p->node;//ptr points on parent, i.e., ptr->c_r = c_p!!!
					T value = ptr->c_r*phi_r_x_r_prime - muBase[p->lambda+s_r];
					lambdaBase[p->lambda+s_r] = value;
				}
			}

			for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p) {
				for (size_t s_r = s_r_e - 1; s_r != 0; --s_r) {
					lambdaBase[p->lambda+s_r] -= lambdaBase[p->lambda];
				}
				lambdaBase[p->lambda] = 0;
			}
		}
	}

	T ReparameterizeRegionAlternateWithLoop(T* lambdaBase, T* muBase, T epsilon, RRegionAlternateWorkspaceID& wspace, typename MPGraph<T, S>::DualWorkspaceID &dw, int num_iters) {
		S* translators;
		std::vector<T> serializedDataFloat;
		std::vector<int> serializedDataInt;
		std::vector<T> potentialData;
		std::vector<int> DataFloatOffsets;
		std::vector<int> DataIntOffsets;
		ReparameterizeRegionAlternateLambda_SerializeData(translators, serializedDataFloat, serializedDataInt, potentialData, DataFloatOffsets, DataIntOffsets);
		int k_e = int(NumberOfRegionsWithParents());
		T dual;
		for(int iter=0;iter<num_iters;++iter) {
			ReparameterizeRegionAlternateMu(lambdaBase, muBase, epsilon, wspace);
			ReparameterizeRegionAlternateLambda_Serialized(lambdaBase, muBase, translators, &serializedDataFloat[0], &serializedDataInt[0], &potentialData[0], &DataFloatOffsets[0], &DataIntOffsets[0], k_e);
			dual = ComputeDual(lambdaBase, epsilon, dw);
			std::cout << iter << ": " << dual << std::endl;
		}
		return dual;
	}

	void ReparameterizeRegionAlternate(T* lambdaBase, T* muBase, T epsilon, RRegionAlternateWorkspaceID& wspace) {
		ReparameterizeRegionAlternateMu(lambdaBase, muBase, epsilon, wspace);
		ReparameterizeRegionAlternateLambda(lambdaBase, muBase);
	}
#endif

	void ReparameterizeRegion(T* lambdaBase, int r, T epsilon, bool additiveUpdate, RRegionWorkspaceID& wspace) {
		MPNode* r_ptr = Graph[ValidRegionMapping[r]];

		size_t ParentLocalIX;
		T* mu_p_r = wspace.RRegionMem;

		T inv_sum_c_p = T(1.)/r_ptr->sum_c_r_c_p;
		//T sum_c_p = r_ptr->sum_c_r_c_p;
		//T sum_c_p = r_ptr->c_r;
		//for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p) {
		//	sum_c_p += p->node->c_r;
		//}

		size_t s_r_e = r_ptr->GetPotentialSize();
		size_t rNumVar = r_ptr->varIX.size();
		//std::vector<S> indivVarStates(rNumVar, 0);
		S* indivVarStates = wspace.IXMem;
		for(S *tmp=indivVarStates, *tmp_e=indivVarStates+rNumVar;tmp!=tmp_e;++tmp) {
			*tmp = 0;
		}
		for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
			T phi_r_x_r_prime = (r_ptr->pot->data == NULL) ? T(0) : r_ptr->weight_r*r_ptr->pot->data[s_r];

			for (typename std::vector<MsgContainer>::const_iterator c = r_ptr->Children.begin(), c_e = r_ptr->Children.end(); c != c_e; ++c) {
				phi_r_x_r_prime += lambdaBase[c->lambda+c->Translator[s_r]];
			}

			ParentLocalIX = 0;
			for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p, ++ParentLocalIX) {
				mu_p_r[ParentLocalIX] = ComputeMu(lambdaBase, p->edge, indivVarStates, rNumVar, epsilon, wspace.MuMem, wspace.MuIXMem);
				phi_r_x_r_prime += mu_p_r[ParentLocalIX];
			}

			//phi_r_x_r_prime /= sum_c_p;
			phi_r_x_r_prime *= inv_sum_c_p;

			ParentLocalIX = 0;
			for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p, ++ParentLocalIX) {
				MPNode* ptr = p->node;//ptr points on parent, i.e., ptr->c_r = c_p!!!
				T value = ptr->c_r*phi_r_x_r_prime - mu_p_r[ParentLocalIX];
				lambdaBase[p->lambda+s_r] = ((additiveUpdate)?value-lambdaBase[p->lambda+s_r]:value);//the employed normalization is commutative
			}

			for (size_t varIX = 0; varIX < rNumVar; ++varIX) {
				++indivVarStates[varIX];
				if (indivVarStates[varIX] == Cardinalities[r_ptr->varIX[varIX]]) {
					indivVarStates[varIX] = 0;
				} else {
					break;
				}
			}
		}

		for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p) {
			for (size_t s_r = s_r_e - 1; s_r != 0; --s_r) {
				lambdaBase[p->lambda+s_r] -= lambdaBase[p->lambda];
			}
			lambdaBase[p->lambda] = 0;
		}
	}

	T ComputeReparameterizationPotential(T* lambdaBase, const MPNode* const r_ptr, const S s_r) const {
		T potVal = ((r_ptr->pot->data != NULL) ? r_ptr->weight_r*r_ptr->pot->data[s_r] : T(0));

		for (typename std::vector<MsgContainer>::const_iterator pn = r_ptr->Parents.begin(), pn_e = r_ptr->Parents.end(); pn != pn_e; ++pn) {
			potVal -= lambdaBase[pn->lambda+s_r];
		}

		for (typename std::vector<MsgContainer>::const_iterator cn = r_ptr->Children.begin(), cn_e = r_ptr->Children.end(); cn != cn_e; ++cn) {
			potVal += lambdaBase[cn->lambda+cn->Translator[s_r]];
		}

		return potVal;
	}

	T ComputeDual(T* lambdaBase, T epsilon, DualWorkspaceID& dw) const {
		T dual = T(0);

		for (typename std::vector<MPNode*>::const_iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
			MPNode* r_ptr = *r;
			T ecr = epsilon*r_ptr->c_r;
			size_t s_r_e = r_ptr->GetPotentialSize();

			T* mem = dw.DualWorkspace;

			T maxVal = -std::numeric_limits<T>::max();
			for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
				T potVal = ComputeReparameterizationPotential(lambdaBase, r_ptr, s_r);

				if (ecr != T(0)) {
					potVal /= ecr;
					mem[s_r] = potVal;
				}

				maxVal = ((potVal > maxVal) ? potVal : maxVal);
			}

			if (ecr != T(0)) {
				T sum = std::exp(mem[0] - maxVal);
				for (size_t s_r = 1; s_r != s_r_e; ++s_r) {
					sum += std::exp(mem[s_r] - maxVal);
				}
				dual += ecr*(maxVal + std::log(sum + T(1e-20)));
			} else {
				dual += maxVal;
			}
		}
		return dual;
	}

	size_t GetLambdaSize() const {
		return LambdaSize;
	}

	void GradientUpdateEdge(T* lambdaBase, int e, T epsilon, T stepSize, bool additiveUpdate, GEdgeWorkspaceID& gew) {
		EdgeID* edge = Edges[e];
		MPNode* r_ptr = edge->childPtr->node;
		MPNode* p_ptr = edge->parentPtr->node;

		size_t s_r_e = r_ptr->GetPotentialSize();
		T* mem_r = gew.mem1;
		ComputeBeliefForRegion(r_ptr, lambdaBase, epsilon, mem_r, s_r_e);
		size_t s_p_e = p_ptr->GetPotentialSize();
		T* mem_p = gew.mem2;
		ComputeBeliefForRegion(p_ptr, lambdaBase, epsilon, mem_p, s_p_e);

		for (size_t s_p = 0; s_p < s_p_e; ++s_p) {
			mem_r[edge->childPtr->Translator[s_p]] -= mem_p[s_p];
		}

		if (additiveUpdate) {
			for (size_t s_r = 0; s_r < s_r_e; ++s_r) {
				//lambdaBase[edge->childPtr->lambda + s_r] += stepSize*mem_r[s_r];//use this line if global memory is identical to local memory
				lambdaBase[edge->childPtr->lambda + s_r] = stepSize*mem_r[s_r];//addition is performed in UpdateEdge
			}
		} else {
			for (size_t s_r = 0; s_r < s_r_e; ++s_r) {
				lambdaBase[edge->childPtr->lambda + s_r] += stepSize*mem_r[s_r];
			}
		}
	}

    void GradStep(T* lambdaBase, T* beliefs, int r, T epsilon, bool additiveUpdate, float lr, DualWorkspaceID& dw) {
		MPNode* r_ptr = Graph[ValidRegionMapping[r]];

		//size_t ParentLocalIX;
		T* grad = dw.DualWorkspace;

		//T sum_c_p = r_ptr->sum_c_r_c_p;
		//T sum_c_p = r_ptr->c_r;
		//for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p) {
		//	sum_c_p += p->node->c_r;
		//}

		size_t s_r_e = r_ptr->GetPotentialSize();
		//size_t rNumVar = r_ptr->varIX.size();
		//std::vector<S> indivVarStates(rNumVar, 0);
		//S* indivVarStates = wspace.IXMem;
		//for(S *tmp=indivVarStates, *tmp_e=indivVarStates+rNumVar;tmp!=tmp_e;++tmp) {
		//	*tmp = 0;
		//}
        size_t s_p_e;
        /*
        ComputeBeliefForRegion(r_ptr, lambdaBase, epsilon, beliefs, s_r_e);
        for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p, ++ParentLocalIX) {
            ComputeBeliefForRegion(p->node, lambdaBase, epsilon, beliefs, s_r_e);
        }*/
                
        for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p) {
            for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
                grad[s_r] = -1*beliefs[r_ptr->belief+s_r];
            }
            s_p_e = p->node->GetPotentialSize();
            for (size_t s_p = 0; s_p != s_p_e; ++s_p) {
                grad[p->Translator[s_p]] += beliefs[p->node->belief+s_p];
			}
            for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
                lambdaBase[p->lambda+s_r] -= lr*grad[s_r];
            }
        }
	}


	//void ComputeBeliefForRegion(MPNode* r_ptr, T* lambdaBase, T epsilon, T* mem, size_t s_r_e, T* belWtPtr, size_t region_ind) {
	void ComputeBeliefForRegion(MPNode* r_ptr, T* lambdaBase, T epsilon, T* mem, size_t s_r_e) {
		T ecr = epsilon*r_ptr->c_r;

		T maxVal = -std::numeric_limits<T>::max();
		for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
			T potVal = ComputeReparameterizationPotential(lambdaBase, r_ptr, s_r);
			if (ecr != T(0)) {
				potVal /= ecr;
			}
			mem[s_r] = potVal;

			maxVal = ((potVal > maxVal) ? potVal : maxVal);
		}
        
		T sum = T(0);
		if (ecr != T(0)) {
			for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
				mem[s_r] = std::exp(mem[s_r] - maxVal);
				sum += mem[s_r];
			}
		} else {
			for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
				mem[s_r] = ((mem[s_r] - maxVal) > -1e-5) ? T(1) : T(0);
				sum += mem[s_r];
			}
		}
        
		for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
			mem[s_r] /= sum;
            entropy -= mem[s_r]*log(mem[s_r]+ std::numeric_limits<T>::epsilon());
            //if (r_ptr->varIX.size() > 1)
            //{
                /*
                if (belWtPtr != NULL) {
                    region_belief_sums[s_r] += mem[s_r]*belWtPtr[region_ind];
                } else {*/
            //        region_belief_sums[s_r] += mem[s_r];
                //}
            //}
		}
	}

	size_t ComputeBeliefSize(bool OnlyUnaries) {
		size_t BeliefSize = 0;
		for (typename std::vector<MPNode*>::const_iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
			MPNode* r_ptr = *r;
			size_t s_r_e = r_ptr->GetPotentialSize();
			size_t numVars = r_ptr->varIX.size();
			if ((OnlyUnaries&&numVars == 1) || !OnlyUnaries) {
				BeliefSize += s_r_e;
			}
		}
		return BeliefSize;
	}
	size_t ComputeBeliefs(T* lambdaBase, T epsilon, T* belPtr, bool OnlyUnaries) {
        T* mem = belPtr;
        entropy = 0;
        size_t belief_offset = 0;
		for (typename std::vector<MPNode*>::const_iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
			MPNode* r_ptr = *r;
			size_t numVars = r_ptr->varIX.size();
			if (OnlyUnaries&&numVars > 1) {
				continue;
			}
			size_t s_r_e = r_ptr->GetPotentialSize();
			ComputeBeliefForRegion(r_ptr, lambdaBase, epsilon, mem, s_r_e);

			r_ptr->belief = belief_offset;
			mem += s_r_e;
            belief_offset += s_r_e;
		}

		return 0;
	}

	void Marginalize(T* curBel, T* oldBel, EdgeID* edge, const std::vector<S>& indivVarStates, T& marg_new, T& marg_old) {
		//MPNode* r_ptr = edge->childPtr->node;
		//MPNode* p_ptr = edge->parentPtr->node;

		size_t numVarsOverlap = indivVarStates.size();
		size_t s_p_stat = 0;
		for (size_t k = 0; k<numVarsOverlap; ++k) {
			s_p_stat += indivVarStates[k]*edge->rStateMultipliers[k];
		}

		size_t s_p_e = edge->newVarSize;
		size_t numVarNew = edge->newVarIX.size();

		//individual vars;
		std::vector<S> indivNewVarStates(numVarNew, 0);
		for (size_t s_p = 0; s_p != s_p_e; ++s_p) {
			size_t s_p_real = s_p_stat;
			for (size_t varIX = 0; varIX<numVarNew; ++varIX) {
				s_p_real += indivNewVarStates[varIX]*edge->newVarStateMultipliers[varIX];
			}

			marg_new += curBel[s_p_real];
			marg_old += oldBel[s_p_real];

			for (size_t varIX = 0; varIX < numVarNew; ++varIX) {
				++indivNewVarStates[varIX];
				if (indivNewVarStates[varIX] == Cardinalities[edge->newVarIX[varIX]]) {
					indivNewVarStates[varIX] = 0;
				} else {
					break;
				}
			}
		}

		/*for(int k=0;k<4;++k) {
			std::cout << curBel[k] << " ";
		}
		std::cout << marg_new << std::endl;
		for(int k=0;k<4;++k) {
			std::cout << oldBel[k] << " ";
		}
		std::cout << marg_old << std::endl;*/
	}

	T ComputeImprovement(T* curBel, T* oldBel) {//not efficient
		T imp = T(0);

		for (typename std::vector<MPNode*>::iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
			MPNode* r_ptr = *r;
			size_t s_r_e = r_ptr->GetPotentialSize();
			size_t belIX_r = ((T*)r_ptr->tmp) - curBel;

			for (typename std::vector<MsgContainer>::const_iterator pn = r_ptr->Parents.begin(), pn_e = r_ptr->Parents.end(); pn != pn_e; ++pn) {
				MPNode* p_ptr = pn->node;
				size_t belIX_p = ((T*)p_ptr->tmp) - curBel;

				T v1 = 0;
				T v2 = 0;
				size_t rNumVar = r_ptr->varIX.size();
				std::vector<S> indivVarStates(rNumVar, 0);
				for(size_t s_r=0;s_r<s_r_e;++s_r) {
					T marg_old = T(0);
					T marg_new = T(0);
					Marginalize(curBel + belIX_p, oldBel + belIX_p, pn->edge, indivVarStates, marg_new, marg_old);

					v1 += curBel[belIX_r+s_r]*std::sqrt(marg_old/oldBel[belIX_r+s_r]);
					v2 += marg_new*std::sqrt(oldBel[belIX_r+s_r]/marg_old);

					for (size_t varIX = 0; varIX < rNumVar; ++varIX) {
						++indivVarStates[varIX];
						if (indivVarStates[varIX] == Cardinalities[r_ptr->varIX[varIX]]) {
							indivVarStates[varIX] = 0;
						} else {
							break;
						}
					}
				}

				imp += std::log(v1) + std::log(v2);
			}
		}

		return imp;
	}

	T getBelief(T* beliefs, S region_num, S val_num) {
        T result = beliefs[Graph[region_num]->belief + ((size_t)val_num)];
 		return result;
	}

	void DeleteBeliefs() {
		MPNode* r_ptr = *Graph.begin();
		delete[]((float*)r_ptr->tmp);
	}
};

//template class MPGraph<float, int>;

#ifdef _MSC_VER
#define likely(x)		(x)
#define unlikely(x)		(x)
#else
#define likely(x)		__builtin_expect(!!(x), 1)
#define unlikely(x)		__builtin_expect(!!(x), 0)
#endif

template <class T, class S>
class ThreadSync {
	enum STATE : unsigned char { NONE, INTR, TERM, INIT};
	STATE state;
	int numThreads;
	T* lambdaGlobal;
	T epsilon;
	MPGraph<T, S>* g;
	int currentlyStoppedThreads;
	std::mutex mtx;
	std::condition_variable cond_;
	typename MPGraph<T, S>::DualWorkspaceID dw;
	CPrecisionTimer CTmr, CTmr1;
	T prevDual;
	std::vector<T> LambdaForNoSync;
public:
	ThreadSync(int nT, T* lambdaGlobal, T epsilon, MPGraph<T, S>* g) : state(NONE), numThreads(nT), lambdaGlobal(lambdaGlobal), epsilon(epsilon), g(g), currentlyStoppedThreads(0), prevDual(std::numeric_limits<T>::max())  {
		dw = g->AllocateDualWorkspaceMem(epsilon);
		state = INIT;
		LambdaForNoSync.assign(g->GetLambdaSize(),0);
	}
	virtual ~ThreadSync() {
		g->DeAllocateDualWorkspaceMem(dw);
	}
	bool checkSync() {
		if (unlikely(state == INTR)) {
			std::unique_lock<std::mutex> lock(mtx);
			if (currentlyStoppedThreads < numThreads - 1) {
				++currentlyStoppedThreads;
				cond_.wait(lock);
			} else if (currentlyStoppedThreads == numThreads - 1) {
				double timeMS = CTmr.Stop()*1000.;
				//std::cout << timeMS << "; " << CTmr1.Stop()*1000. << "; ";
				T dualVal = g->ComputeDual(lambdaGlobal, epsilon, dw);
				//std::cout << timeMS <<"; " << CTmr1.Stop()*1000. << "; " << std::setprecision(15) << dualVal << std::endl;
				std::cout << timeMS <<"; " << CTmr1.Stop()*1000. << "; " << dualVal << std::endl;
				//if (dualVal>prevDual) {
				//	std::cout << " (*)";
				//}
				//std::cout << std::endl;
				prevDual = dualVal;
				CTmr.Start();
				state = NONE;
				currentlyStoppedThreads = 0;
				cond_.notify_all();
			}
		} else if (unlikely(state == TERM)) {
			std::unique_lock<std::mutex> lock(mtx);
			++currentlyStoppedThreads;
			if (currentlyStoppedThreads == numThreads) {
				//T dualVal = g->ComputeDual(lambdaGlobal, epsilon, dw);
				//std::cout << std::setprecision(15) << dualVal << std::endl;
				std::cout << "All threads terminated." << std::endl;
				state = NONE;
				currentlyStoppedThreads = 0;
				cond_.notify_all();
			}
			return false;
		} else if (unlikely(state == INIT)) {
			std::unique_lock<std::mutex> lock(mtx);
			++currentlyStoppedThreads;
			if (currentlyStoppedThreads == numThreads) {
				std::cout << "All threads running." << std::endl;
			}
			cond_.wait(lock);
		}
		return true;
	}
	void interruptFunc() {
		std::unique_lock<std::mutex> lock(mtx);
		state = INTR;
		cond_.wait(lock);
	}
	void terminateFunc() {
		std::unique_lock<std::mutex> lock(mtx);
		state = TERM;
		cond_.wait(lock);
	}
	void ComputeDualNoSync() {
		double timeMS = CTmr.Stop()*1000.;
		std::copy(lambdaGlobal, lambdaGlobal+LambdaForNoSync.size(), &LambdaForNoSync[0]);
		T dualVal = g->ComputeDual(&LambdaForNoSync[0], epsilon, dw);
		std::cout << timeMS <<"; " << CTmr1.Stop()*1000. << "; " << dualVal << std::endl;
	}
	bool startFunc() {
		std::unique_lock<std::mutex> lock(mtx);
		if (currentlyStoppedThreads == numThreads) {
			state = NONE;
			currentlyStoppedThreads = 0;
			CTmr1.Start();
			CTmr.Start();
			cond_.notify_all();
			return true;
		} else {
			return false;
		}
	}
};

//#define WHICH_FUNC 1

//template class ThreadSync<float, int>;

template <class T, class S>
class ThreadWorker {
	size_t cnt;
	ThreadSync<T, S>* ts;
	std::thread* thread_;
	MPGraph<T, S>* g;
	T epsilon;
	int randomSeed;
	T* lambdaGlobal;
	T* lambdaBase;
	size_t msgSize;
	std::vector<T> lambdaLocal;

#if WHICH_FUNC==1
	typename MPGraph<T, S>::RRegionWorkspaceID rrw;
#elif WHICH_FUNC==2
	typename MPGraph<T, S>::REdgeWorkspaceID rew;
#elif WHICH_FUNC==3
	typename MPGraph<T, S>::FunctionUpdateWorkspaceID fw;
#else
	#error No appropriate function defined.
#endif

	std::uniform_int_distribution<int>* uid;
	std::mt19937 eng;
	T* stepsize;
public:
	ThreadWorker(ThreadSync<T, S>* ts, MPGraph<T, S>* g, T epsilon, int randomSeed, T* lambdaGlobal, T* stepsize) : cnt(0), ts(ts), thread_(NULL), g(g), epsilon(epsilon), randomSeed(randomSeed), lambdaGlobal(lambdaGlobal), stepsize(stepsize) {
		msgSize = g->GetLambdaSize();
		assert(msgSize > 0);
		lambdaLocal.assign(msgSize, T(0));
		lambdaBase = &lambdaLocal[0];

#if WHICH_FUNC==1
		rrw = g->AllocateReparameterizeRegionWorkspaceMem(epsilon);
		uid = new std::uniform_int_distribution<int>(0, g->NumberOfRegionsWithParents() - 1);
#elif WHICH_FUNC==2
		rew = g->AllocateReparameterizeEdgeWorkspaceMem(epsilon);
		uid = new std::uniform_int_distribution<int>(0, g->NumberOfEdges() - 1);
#elif WHICH_FUNC==3
		fw = g->AllocateFunctionUpdateWorkspaceMem();
		uid = new std::uniform_int_distribution<int>(0, g->NumberOfRegionsTotal()-1);
#endif

		eng.seed(randomSeed);
	}
	~ThreadWorker() {
#if WHICH_FUNC==1
		g->DeAllocateReparameterizeRegionWorkspaceMem(rrw);
#elif WHICH_FUNC==2
		g->DeAllocateReparameterizeEdgeWorkspaceMem(rew);
#elif WHICH_FUNC==3
		g->DeAllocateFunctionUpdateWorkspaceID(fw);
#endif
		if (uid != NULL) {
			delete uid;
			uid = NULL;
		}
		if (thread_ != NULL) {
			delete thread_;
			thread_ = NULL;
		}
	}
	void start() {
		thread_ = new std::thread(std::bind(&ThreadWorker::run, this));
		thread_->detach();
	}
	size_t GetCount() {
		return cnt;
	}
private:
	void run() {
		std::cout << "Thread started" << std::endl;
		//typename MPGraph<T, S>::DualWorkspaceID dw;
		//dw = g->AllocateDualWorkspaceMem(epsilon);
		while (ts->checkSync()) {
			//int ix = (*uid)(eng);
			//int ix = (randomSeed == 0) ? 0 : 1;

			//std::copy(lambdaGlobal, lambdaGlobal + msgSize, lambdaBase);
			//g->CopyMessagesForEdge(lambdaGlobal, lambdaBase, ix);

			//std::this_thread::sleep_for(std::chrono::milliseconds(500 + randomSeed*100000));

			//std::cout << randomSeed << ": " << ix << ": " << std::endl;
			//for (size_t k = 0; k < msgSize; ++k) {
			//	printf("%.3f ", lambdaBase[k]);
			//}
			//T dualValOrig = g->ComputeDual(lambdaBase, epsilon, dw);
			//std::cout << " --> " << std::setprecision(15) << dualValOrig << std::endl;

			//T* belPtr1 = NULL;
			//size_t belSize = g->ComputeBeliefs(lambdaBase, epsilon, &belPtr1, false);
			//std::cout << "(";
			//for (size_t k=0;k<belSize;++k) {
			//	printf("%.3f ", belPtr1[k]);
			//}
			//std::cout << ")" << std::endl;

			//std::cout << "Improvement: " << g->ComputeImprovement(belPtr1, belPtr1)/6 << std::endl;

			//g->ReparameterizeEdge(lambdaBase, ix, epsilon, false, rew);
			//for (size_t k = 0; k < msgSize; ++k) {
			//	printf("%.3f ", lambdaBase[k]);
			//}
			//T dualValBase = g->ComputeDual(lambdaBase, epsilon, dw);
			//std::cout << " --> " << std::setprecision(15) << dualValBase  <<  " Diff: " << dualValOrig - dualValBase << std::endl;

			//T* belPtr2 = NULL;
			//belSize = g->ComputeBeliefs(lambdaBase, epsilon, &belPtr2, false);
			//std::cout << "(";
			//for (size_t k=0;k<belSize;++k) {
			//	printf("%.3f ", belPtr2[k]);
			//}
			//std::cout << ")" << std::endl;

			//g->UpdateEdge(lambdaBase, lambdaGlobal, ix, false);
			//for (size_t k = 0; k < msgSize; ++k) {
			//	printf("%.3f ", lambdaGlobal[k]);
			//}
			//T dualValGlob = g->ComputeDual(lambdaGlobal, epsilon, dw);
			//std::cout << " --> " << std::setprecision(15) << dualValGlob;
			//if (dualValBase < dualValGlob) {
			//	std::cout << " (*)";
			//}
			//std::cout << std::endl;

			//T* belPtr3 = NULL;
			//belSize = g->ComputeBeliefs(lambdaGlobal, epsilon, &belPtr3, false);
			//std::cout << "(";
			//for (size_t k=0;k<belSize;++k) {
			//	printf("%.3f ", belPtr3[k]);
			//}
			//std::cout << ")" << std::endl;

			//delete[] belPtr1;
			//delete[] belPtr2;
			//delete[] belPtr3;

#if WHICH_FUNC==1
			int ix = (*uid)(eng);
			g->CopyMessagesForStar(lambdaGlobal, lambdaBase, ix);
			g->ReparameterizeRegion(lambdaBase, ix, epsilon, false, rrw);
			g->UpdateRegion(lambdaBase, lambdaGlobal, ix, false);

			/*ix = (*uid)(eng);
			g->CopyMessagesForStar(lambdaGlobal, lambdaBase, ix);
			g->ReparameterizeRegion(lambdaBase, ix, epsilon, false, rrw);
			g->UpdateRegion(lambdaBase, lambdaGlobal, ix, false);*/
#elif WHICH_FUNC==2
			int ix = (*uid)(eng);
			g->CopyMessagesForEdge(lambdaGlobal, lambdaBase, ix);
			g->ReparameterizeEdge(lambdaBase, ix, epsilon, false, rew);
			g->UpdateEdge(lambdaBase, lambdaGlobal, ix, false);
#elif WHICH_FUNC==3
			int ix = (*uid)(eng);
			g->CopyMessagesForLocalFunction(lambdaGlobal, lambdaBase, ix);
			g->ComputeLocalFunctionUpdate(lambdaBase, ix, epsilon, *stepsize, true, fw);
			g->UpdateLocalFunction(lambdaBase, lambdaGlobal, ix, true);
			//if(randomSeed==0 && (cnt+1)%100000==0) {
			//	*stepsize *= 0.9;
			//}
#endif
      //std::this_thread::sleep_for(std::chrono::milliseconds(2));
			++cnt;
		}
		//g->DeAllocateDualWorkspaceMem(dw);
	}
};

//template class ThreadWorker<float, int>;

template <typename T, typename S>
class AsyncRMPThread {
	std::vector<T> lambdaGlobal;
public:
	int RunMP(MPGraph<T, S>& g, T epsilon, int numIterations, int numThreads, int WaitTimeInMS) {
		size_t msgSize = g.GetLambdaSize();
		if (msgSize == 0) {
			typename MPGraph<T, S>::DualWorkspaceID dw = g.AllocateDualWorkspaceMem(epsilon);
			std::cout << "0: " << g.ComputeDual(NULL, epsilon, dw) << std::endl;
			g.DeAllocateDualWorkspaceMem(dw);
			return 0;
		}

		std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1);

		lambdaGlobal.assign(msgSize, T(0));
		T* lambdaGlob = &lambdaGlobal[0];

		ThreadSync<T, S> sy(numThreads, lambdaGlob, epsilon, &g);

		T stepsize = -0.1;
		std::vector<ThreadWorker<T, S>*> ex(numThreads, NULL);
		for (int k = 0; k < numThreads; ++k) {
			ex[k] = new ThreadWorker<T, S>(&sy, &g, epsilon, k, lambdaGlob, &stepsize);
		}
		for (int k = 0; k < numThreads; ++k) {
			ex[k]->start();
		}

		while (!sy.startFunc()) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		}
		for (int k = 0; k < numIterations; ++k) {
			std::this_thread::sleep_for(std::chrono::milliseconds(WaitTimeInMS));
			//sy.interruptFunc();
			sy.ComputeDualNoSync();
		}
		sy.terminateFunc();

		size_t regionUpdates = 0;
		for(int k=0;k<numThreads;++k) {
			size_t tmp = ex[k]->GetCount();
			std::cout << "Thread " << k << ": " << tmp << std::endl;
			regionUpdates += tmp;
			delete ex[k];
		}
		std::cout << "Region updates: " << regionUpdates << std::endl;
		std::cout << "Total regions:  " << g.NumberOfRegionsWithParents() << std::endl;

		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		std::cout << "Terminating program." << std::endl;
		return 0;
	}

	size_t GetBeliefs(MPGraph<T, S>& g, T epsilon, T* belPtr, bool OnlyUnaries) {
		size_t msgSize = g.GetLambdaSize();
		if (msgSize == 0) {
			return g.ComputeBeliefs(NULL, epsilon, belPtr, OnlyUnaries);
		} else {
			if (lambdaGlobal.size() != msgSize) {
				std::cout << "Message size does not fit requirement. Reassigning." << std::endl;
				lambdaGlobal.assign(msgSize, T(0));
			}
			return g.ComputeBeliefs(&lambdaGlobal[0], epsilon, belPtr, OnlyUnaries);
		}
		return size_t(-1);
	}
};

template <typename T, typename S>
class RMP_Grad{
public:
        MPGraph<T, S>& g;
	RMP_Grad(MPGraph<T, S>& graph) : g(graph)  {
	};
	virtual ~RMP_Grad() {
	    //std::cout << "\t\tTHE BELIEFS ARE BEING DESTROYED RIGHT NOW" << std::endl;
            //delete beliefsGlobal;
	};

	T GradStep(T* lambdaGlobal, T* beliefsGlobal, T epsilon, int num_iters, T lr, typename MPGraph<T, S>::DualWorkspaceID &dw) {
		size_t msgSize = g.GetLambdaSize();
		if (msgSize == 0) {
			//typename MPGraph<T, S>::DualWorkspaceID dw = g.AllocateDualWorkspaceMem(epsilon);
			//std::cout << "0: " << g.ComputeDual(NULL, epsilon, dw) << std::endl;
			//g.DeAllocateDualWorkspaceMem(dw);
			return 0;
		}

		//std::vector<T> lambdaGlobal(msgSize, T(0));

		std::cout << std::setprecision(15);

		//typename MPGraph<T, S>::DualWorkspaceID dw = g.AllocateDualWorkspaceMem(1.0); //We need to force allocation of memory
		//typename MPGraph<T, S>::RRegionWorkspaceID rrw = g.AllocateReparameterizeRegionWorkspaceMem(epsilon);
		//typename MPGraph<T, S>::REdgeWorkspaceID rew = g.AllocateReparameterizeEdgeWorkspaceMem(epsilon);
		for (int iter = 0; iter < num_iters; ++iter) {
            g.ComputeBeliefs(lambdaGlobal, epsilon, beliefsGlobal, false);
			for (int k = 0; k < int(g.NumberOfRegionsWithParents()); ++k) {
				g.GradStep(lambdaGlobal, beliefsGlobal, k, epsilon, false, lr, dw);
			}
			//std::cout << iter << ": " << g.ComputeDual(&lambdaGlobal[0], epsilon, dw) << std::endl;
		}
		T result = g.ComputeDual(lambdaGlobal, epsilon, dw);
		//g.DeAllocateReparameterizeRegionWorkspaceMem(rrw);
		//g.DeAllocateReparameterizeEdgeWorkspaceMem(rew);
		//g.DeAllocateDualWorkspaceMem(dw);
		return result;
	}
	/*
	size_t UpdateBeliefs(T epsilon, bool OnlyUnaries) {
		size_t msgSize = g.GetLambdaSize();
		if (msgSize == 0) {
			return g.ComputeBeliefs(NULL, epsilon, &beliefsGlobal, OnlyUnaries);
		} else {
			if (lambdaGlobal.size() != msgSize) {
				std::cout << "Message size does not fit requirement. Reassigning." << std::endl;
				lambdaGlobal.assign(msgSize, T(0));
			}
			return g.ComputeBeliefs(&lambdaGlobal[0], epsilon, &beliefsGlobal, OnlyUnaries);
		}
		return size_t(-1);
	}*/

	T getBelief(S region_num, S val_ind) {
		
		//std::cout << "GETTING BELIEF" << std::endl;
		return g.getBelief(region_num, val_ind);
        }
};



template <typename T, typename S>
class RMP{
public:
        MPGraph<T, S>& g;
        T* beliefsGlobal;
	RMP(MPGraph<T, S>& graph) : g(graph), beliefsGlobal(NULL) {
	};
	virtual ~RMP() {
	    //std::cout << "\t\tTHE BELIEFS ARE BEING DESTROYED RIGHT NOW" << std::endl;
            //delete beliefsGlobal;
	};

	T RunMP(T* lambdaGlobal, T epsilon, int num_iters, typename MPGraph<T, S>::DualWorkspaceID &dw, typename MPGraph<T, S>::RRegionWorkspaceID &rrw) {
		size_t msgSize = g.GetLambdaSize();
		if (msgSize == 0) {
			//std::cout << "0: " << g.ComputeDual(NULL, epsilon, dw) << std::endl;
			return 0;
		}

		//std::vector<T> lambdaGlobal(msgSize, T(0));

		std::cout << std::setprecision(15);

		//typename MPGraph<T, S>::DualWorkspaceID dw = g.AllocateDualWorkspaceMem(epsilon);
		//typename MPGraph<T, S>::RRegionWorkspaceID rrw = g.AllocateReparameterizeRegionWorkspaceMem(epsilon);
		//typename MPGraph<T, S>::REdgeWorkspaceID rew = g.AllocateReparameterizeEdgeWorkspaceMem(epsilon);
		T result = 0;
		for (int iter = 0; iter < num_iters; ++iter) {
			//for (int k = 0; k < int(g.NumberOfEdges()); ++k) {
			//	g.ReparameterizeEdge(&lambdaGlobal[0], k, epsilon, false, rew);
			//}
			for (int k = 0; k < int(g.NumberOfRegionsWithParents()); ++k) {
				g.ReparameterizeRegion(lambdaGlobal, k, epsilon, false, rrw);
			}
			result = g.ComputeDual(lambdaGlobal, epsilon, dw);
			std::cout << iter << ": " << result << std::endl;
		}
		//T result = g.ComputeDual(lambdaGlobal, epsilon, dw);
		//g.DeAllocateReparameterizeRegionWorkspaceMem(rrw);
		//g.DeAllocateReparameterizeEdgeWorkspaceMem(rew);
		//g.DeAllocateDualWorkspaceMem(dw);
		return result;
	}
	/*
	size_t UpdateBeliefs(T epsilon, bool OnlyUnaries) {
		size_t msgSize = g.GetLambdaSize();
		if (msgSize == 0) {
			return g.ComputeBeliefs(NULL, epsilon, &beliefsGlobal, OnlyUnaries);
		} else {
			if (lambdaGlobal.size() != msgSize) {
				std::cout << "Message size does not fit requirement. Reassigning." << std::endl;
				lambdaGlobal.assign(msgSize, T(0));
			}
			return g.ComputeBeliefs(&lambdaGlobal[0], epsilon, &beliefsGlobal, OnlyUnaries);
		}
		return size_t(-1);
	}*/

	T getBelief(S region_num, S val_ind) {
		
		//std::cout << "GETTING BELIEF" << std::endl;
		return g.getBelief(region_num, val_ind);
        }
};

#ifdef WITH_ALTERNATING
template <typename T, typename S>
class RMPAlternate{
public:
        MPGraph<T, S>& g;
        T* beliefsGlobal;
	RMPAlternate(MPGraph<T, S>& graph) : g(graph), beliefsGlobal(NULL) {
	};
	virtual ~RMPAlternate() {
	    //std::cout << "\t\tTHE BELIEFS ARE BEING DESTROYED RIGHT NOW" << std::endl;
            //delete beliefsGlobal;
	};

	T RunMP(T* lambdaGlobal, T* muGlobal, T epsilon, int num_iters, typename MPGraph<T, S>::DualWorkspaceID &dw, typename MPGraph<T, S>::RRegionAlternateWorkspaceID &rraw) {
		size_t msgSize = g.GetLambdaSize();
		if (msgSize == 0) {
			//std::cout << "0: " << g.ComputeDual(NULL, epsilon, dw) << std::endl;
			return 0;
		}

		//std::vector<T> lambdaGlobal(msgSize, T(0));

		std::cout << std::setprecision(15);

		//typename MPGraph<T, S>::DualWorkspaceID dw = g.AllocateDualWorkspaceMem(epsilon);
		//typename MPGraph<T, S>::RRegionWorkspaceID rrw = g.AllocateReparameterizeRegionWorkspaceMem(epsilon);
		//typename MPGraph<T, S>::REdgeWorkspaceID rew = g.AllocateReparameterizeEdgeWorkspaceMem(epsilon);
/*		for (int iter = 0; iter < num_iters; ++iter) {
			//for (int k = 0; k < int(g.NumberOfEdges()); ++k) {
			//	g.ReparameterizeEdge(&lambdaGlobal[0], k, epsilon, false, rew);
			//}
			g.ReparameterizeRegionAlternate(lambdaGlobal, muGlobal, epsilon, rraw);
			std::cout << iter << ": " << g.ComputeDual(lambdaGlobal, epsilon, dw) << std::endl;
		}
		T result = g.ComputeDual(lambdaGlobal, epsilon, dw);
*/		//g.DeAllocateReparameterizeRegionWorkspaceMem(rrw);
		//g.DeAllocateReparameterizeEdgeWorkspaceMem(rew);
		//g.DeAllocateDualWorkspaceMem(dw);
		T result = g.ReparameterizeRegionAlternateWithLoop(lambdaGlobal, muGlobal, epsilon, rraw, dw, num_iters);
		return result;
	}

	T getBelief(S region_num, S val_ind) {
		
		//std::cout << "GETTING BELIEF" << std::endl;
		return g.getBelief(region_num, val_ind);
        }
};
#endif

#ifdef ASYNC_RMP_OPENMP

template <typename T, typename S>
class AsyncRMP {
	std::vector<T> lambdaGlobal;
public:
	AsyncRMP() {};
	virtual ~AsyncRMP() {};

	int RunMP(MPGraph<T, S>& g, T epsilon) {
		size_t msgSize = g.GetLambdaSize();
		if (msgSize == 0) {
			typename MPGraph<T, S>::DualWorkspaceID dw = g.AllocateDualWorkspaceMem(epsilon);
			std::cout << "0: " << g.ComputeDual(NULL, epsilon, dw) << std::endl;
			g.DeAllocateDualWorkspaceMem(dw);
			return 0;
		}



		//g.AllocateDualWorkspaceMem(epsilon);

		lambdaGlobal.assign(msgSize, T(0));
		T* lambdaGlob = (msgSize > 0) ? &lambdaGlobal[0] : NULL;

		omp_set_num_threads(16);

		CPrecisionTimer CTmrT, CTmr;
		double CopyTime = 0;
		CTmrT.Start();

#pragma omp parallel
		{
			int threadID = omp_get_thread_num();
			std::mt19937 eng(threadID);
			std::vector<T> lambda(msgSize, T(0));
			T* lambdaBase = (msgSize > 0) ? &lambda[0] : NULL;
			typename MPGraph<T, S>::DualWorkspaceID dw = g.AllocateDualWorkspaceMem(epsilon);

			/*typename MPGraph<T, S>::REdgeWorkspaceID rew = g.AllocateReparameterizeEdgeWorkspaceMem(epsilon);
			std::uniform_int_distribution<int> uid(0, g.NumberOfEdges() - 1);
			for (int iter = 0; iter < 100; ++iter) {
				int ix = uid(eng);
				//std::copy(lambdaGlobal.begin(), lambdaGlobal.end(), lambda.begin());
				g.CopyMessagesForEdge(lambdaGlob, lambdaBase, ix);
				g.ReparameterizeEdge(lambdaBase, ix, epsilon, true, rew);
				g.UpdateEdge(lambdaBase, lambdaGlob, ix, true);
				std::cout << iter << ": " << g.ComputeDual(lambdaGlob, epsilon, dw) << std::endl;
			}
			g.DeAllocateReparameterizeEdgeWorkspaceMem(rew);*/

			typename MPGraph<T, S>::RRegionWorkspaceID rrw = g.AllocateReparameterizeRegionWorkspaceMem(epsilon);
			std::uniform_int_distribution<int> uid(0, g.NumberOfRegions() - 1);
			for (int iter = 0; iter < 20; ++iter) {
				for (int k = 0; k < g.NumberOfRegions(); ++k) {
					int ix = uid(eng);//k;//uid(eng);
					CTmr.Start();
					//std::copy(lambdaGlobal.begin(), lambdaGlobal.end(), lambda.begin());
					g.CopyMessagesForStar(lambdaGlob, lambdaBase, ix);
					CopyTime += CTmr.Stop();
					g.ReparameterizeRegion(lambdaBase, ix, epsilon, false, rrw);
					g.UpdateRegion(lambdaBase, lambdaGlob, ix, false);
				}
#pragma omp barrier
				if (threadID == 0) {
					std::cout << iter << ": " << g.ComputeDual(lambdaGlob, epsilon, dw) << std::endl;
				}
#pragma omp barrier
			}
			g.DeAllocateReparameterizeRegionWorkspaceMem(rrw);

			/*typename MPGraph<T, S>::GEdgeWorkspaceID gew = g.AllocateGradientEdgeWorkspaceMem();
			std::uniform_int_distribution<int> uid(0, g.NumberOfEdges() - 1);
			for (int iter = 0; iter < 100; ++iter) {
				int ix = uid(eng);
				//std::copy(lambdaGlobal.begin(), lambdaGlobal.end(), lambda.begin());
				g.CopyMessagesForEdge(lambdaGlob, lambdaBase, ix);
				g.GradientUpdateEdge(lambdaBase, ix, epsilon, T(1.0), false, gew);
				g.UpdateEdge(lambdaBase, lambdaGlob, ix, false);
				T dual = g.ComputeDual(lambdaGlob, epsilon, dw);
				std::cout << iter << ": " << dual << std::endl;
			}
			g.DeAllocateGradientEdgeWorkspaceMem(gew);*/

			g.DeAllocateDualWorkspaceMem(dw);
		}
		std::cout << "Copy Time:  " << CopyTime << std::endl;
		std::cout << "Total Time: " << CTmrT.Stop() << std::endl;
		return 0;
	}

	size_t GetBeliefs(MPGraph<T, S>& g, T epsilon, T** belPtr, bool OnlyUnaries) {
		size_t msgSize = g.GetLambdaSize();
		if (msgSize == 0) {
			return g.ComputeBeliefs(NULL, epsilon, belPtr, OnlyUnaries);
		} else {
			if (lambdaGlobal.size() != msgSize) {
				std::cout << "Message size does not fit requirement. Reassigning." << std::endl;
				lambdaGlobal.assign(msgSize, T(0));
			}
			return g.ComputeBeliefs(&lambdaGlobal[0], epsilon, belPtr, OnlyUnaries);
		}
		return size_t(-1);
	}
};

#endif

#endif /* H_REGION */
