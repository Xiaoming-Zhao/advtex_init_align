/*
The MIT License (MIT)

Copyright (c) 2017 Rohan Sawhney and Keenan Crane 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include "MeshData.h"

namespace bff {

class Generators {
public:
	// computes generators
	static void compute(Mesh& mesh);

private:
	// builds primal spanning tree
	static void buildPrimalSpanningTree(Mesh& mesh,
										VertexData<VertexCIter>& primalParent);

	// checks whether an edge is in the primal spanning tree
	static bool inPrimalSpanningTree(EdgeCIter e,
									 const VertexData<VertexCIter>& primalParent);

	// builds dual spanning tree
	static void buildDualSpanningTree(Mesh& mesh,
									  FaceData<FaceCIter>& ngon,
									  FaceData<FaceCIter>& dualParent,
									  const VertexData<VertexCIter>& primalParent);

	// checks whether an edge is in the dual spanning tree
	static bool inDualSpanningTree(EdgeCIter e,
								   const FaceData<FaceCIter>& ngon,
								   const FaceData<FaceCIter>& dualParent);

	// returns shared edge between u and v
	static EdgeIter sharedEdge(VertexCIter u, VertexCIter v);

	// creates boundary from generators
	static void createBoundary(Mesh& mesh);
};

} // namespace bff
