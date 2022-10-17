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

class Face {
public:
	// constructor
	Face(Mesh *mesh);

	// copy constructor
	Face(const Face& f);

	// one of the halfedges associated with this face
	HalfEdgeIter halfEdge() const;

	// sets halfedge
	void setHalfEdge(HalfEdgeCIter he);

	// sets mesh
	void setMesh(Mesh *mesh);

	// returns face normal
	Vector normal(bool normalize = true) const;

	// returns face area
	double area() const;

	// returns face centroid in uv plane
	Vector centroidUV() const;

	// returns face area in uv plane
	double areaUV() const;

	// checks if this face is real
	bool isReal() const;

	// flag to indicate whether this face fills a hole
	bool fillsHole;

	// flag to indicate whether this face is incident to the north
	// pole of a stereographic projection from the disk to a sphere
	bool inNorthPoleVicinity;

	// id between 0 and |F|-1
	int index;

private:
	// index of one of the halfedges associated with this face
	int halfEdgeIndex;

	// pointer to mesh this face belongs to
	Mesh *mesh;
};

} // namespace bff
