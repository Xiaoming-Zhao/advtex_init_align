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

class Vertex {
public:
	// constructor
	Vertex(Mesh *mesh);

	// copy constructor
	Vertex(const Vertex& v);

	// returns one of the halfedges associated with this vertex
	HalfEdgeIter halfEdge() const;

	// sets halfedge
	void setHalfEdge(HalfEdgeCIter he);

	// returns one of the wedges (a.k.a. corner) associated with this vertex
	WedgeIter wedge() const;

	// sets mesh
	void setMesh(Mesh *mesh);

	// checks if this vertex is on the boundary
	bool onBoundary(bool checkIfOnCut = true) const;

	// checks if this vertex is isolated
	bool isIsolated() const;

	// returns degree
	int degree() const;
	int degreeDebug() const;

	// returns angle weighted average of adjacent face normals
	Vector normal() const;

	// returns 2π minus sum of incident angles. Note: only valid for interior vertices
	double angleDefect() const;

	// returns exterior angle. Note: only valid for boundary vertices
	double exteriorAngle() const;

	// position
	Vector position;

	// flag to indicate whether this vertex is a neighbor of or is
	// the north pole of a stereographic projection from the disk to a sphere
	bool inNorthPoleVicinity;

	// id between 0 and |V|-1
	int index;

	// id of the reference vertex this vertex is a duplicate of
	int referenceIndex;

private:
	// index of one of the halfedges associated with this vertex
	int halfEdgeIndex;

	// pointer to mesh this vertex belongs to
	Mesh *mesh;
};

} // namespace bff
