/*
The MIT License (MIT)

Copyright (c) 2017 Rohan Sawhney and Keenan Crane 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include "Types.h"

namespace bff {

class Edge {
public:
	// constructor
	Edge(Mesh *mesh);

	// copy constructor
	Edge(const Edge& e);

	// returns one of the halfedges associated with this edge
	HalfEdgeIter halfEdge() const;

	// sets halfedge
	void setHalfEdge(HalfEdgeCIter he);

	// sets mesh
	void setMesh(Mesh *mesh);

	// returns edge length
	double length() const;

	// returns cotan weight associated with this edge
	double cotan() const;

	// checks if this edge is on the boundary
	bool onBoundary() const;

	// boolean flag to indicate if edge is on a generator
	bool onGenerator;

	// boolean flag to indicate if edge is on a cut
	bool onCut;

	// boolean flag to indicate if cut can pass through edge
	bool isCuttable;

	// id between 0 and |E|-1
	int index;

private:
	// index of one of the halfedges associated with this edge
	int halfEdgeIndex;

	// pointer to mesh this edge belongs to
	Mesh *mesh;
};

} // namespace bff
