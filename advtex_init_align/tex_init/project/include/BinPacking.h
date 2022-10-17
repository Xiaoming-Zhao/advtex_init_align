/*
The MIT License (MIT)

Copyright (c) 2017 Rohan Sawhney and Keenan Crane 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

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
