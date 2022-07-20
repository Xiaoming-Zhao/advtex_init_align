#pragma once

#include "Mesh.h"
#include "Rect.h"

namespace bff {

class BinPacking {
public:
    static bool attemptPacking(int boxLength, double unitsPerInt, const std::vector<rbp::Rect>& rectangles,
					           std::vector<Vector>& newCenters, std::vector<bool>& flippedBins,
					           Vector& modelMinBounds, Vector& modelMaxBounds);
	// packs uvs
	static void pack(Model& model, const std::vector<bool>& mappedToSphere,
					 std::vector<Vector>& originalCenters, std::vector<Vector>& newCenters,
					 std::vector<bool>& flippedBins, Vector& modelMinBounds, Vector& modelMaxBounds);
};

} // namespace bff
