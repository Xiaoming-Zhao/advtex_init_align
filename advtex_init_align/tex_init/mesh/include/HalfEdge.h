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

class HalfEdge {
public:
	// constructor
	HalfEdge(Mesh *mesh);

	// copy constructor
	HalfEdge(const HalfEdge& he);

	// returns the next halfedge (in CCW order) associated with this halfedge's face
	HalfEdgeIter next() const;

	// returns the prev halfedge associated with this halfedge's face
	HalfEdgeIter prev() const;

	// returns the other halfedge associated with this halfedge's edge
	HalfEdgeIter flip() const;

	// returns the vertex at the base of this halfedge
	VertexIter vertex() const;

	// returns the edge associated with this halfedge
	EdgeIter edge() const;

	// returns the face associated with this halfedge
	FaceIter face() const;

	// returns the corner opposite to this halfedge. Undefined if this halfedge
	// is on the boundary
	CornerIter corner() const;

	// returns the wedge (a.k.a. corner) associated with this halfedge
	WedgeIter wedge() const;

	// sets next halfedge
	void setNext(HalfEdgeCIter he);

	// sets prev halfedge
	void setPrev(HalfEdgeCIter he);

	// sets flip halfedge
	void setFlip(HalfEdgeCIter he);

	// sets vertex
	void setVertex(VertexCIter v);

	// sets edge
	void setEdge(EdgeCIter e);

	// sets face
	void setFace(FaceCIter f);

	// sets corner
	void setCorner(CornerCIter c);

	// sets mesh
	void setMesh(Mesh *mesh);

	// returns the cotan weight associated with this halfedge
	double cotan() const;

	// boolean flag to indicate if halfedge is on the boundary
	bool onBoundary;

	// id between 0 and |H|-1
	int index;

private:
	// indices of adjacent mesh elements
	int nextIndex;
	int prevIndex;
	int flipIndex;
	int vertexIndex;
	int edgeIndex;
	int faceIndex;
	int cornerIndex;

	// pointer to mesh this halfedge belongs to
	Mesh *mesh;
};

} // namespace bff
