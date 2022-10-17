/*
The MIT License (MIT)

Copyright (c) 2017 Rohan Sawhney and Keenan Crane 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include "Mesh.h"

namespace bff {

class HoleFiller {
public:
	// fills holes. When fillAll is false, all holes except the longest are filled.
	// However, the length of the longest hole must be larger than 50% of the mesh
	// diameter for it not to be filled. Returns true if all holes are filled and
	// false otherwise.
	static bool fill(Mesh& mesh, bool fillAll = false);

private:
	// returns index of longest boundary loop
	static int longestBoundaryLoop(double& loopLength,
								   std::vector<std::vector<HalfEdgeIter>>& boundaryHalfEdges,
								   const std::vector<Face>& boundaries);

	// fill hole
	static void fill(const std::vector<HalfEdgeIter>& boundaryHalfEdges, Mesh& mesh);
};

} // namespace bff
