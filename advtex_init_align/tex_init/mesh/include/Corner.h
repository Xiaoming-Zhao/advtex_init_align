/*
The MIT License (MIT)

Copyright (c) 2017 Rohan Sawhney and Keenan Crane 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include "Types.h"
#include "Spline.h"

namespace bff {

class Corner {
public:
	// constructor
	Corner(Mesh *mesh);

	// copy constructor
	Corner(const Corner& c);

	// returns the halfedge opposite to this corner
	HalfEdgeIter halfEdge() const;

	// sets halfedge
	void setHalfEdge(HalfEdgeCIter he);

	// returns the vertex associated with this corner
	VertexIter vertex() const;

	// returns the face associated with this corner
	FaceIter face() const;

	// returns the next corner (in CCW order) associated with this corner's face
	CornerIter next() const;

	// returns the previous corner (in CCW order) associated with this corner's face
	CornerIter prev() const;

	// returns the next wedge (in CCW order) along the (possibly cut) boundary
	WedgeIter nextWedge() const;

	// sets mesh
	void setMesh(Mesh *mesh);

	// returns the angle (in radians) at this corner
	double angle() const;

	// checks if this corner is real
	bool isReal() const;

	// Note: wedge must lie on the (possibly cut) boundary
	// computes the exterior angle at this wedge
	double exteriorAngle() const;

	// computes the scaling at this wedge
	double scaling() const;

	// computes the tangent at this wedge
	Vector tangent() const;

	// uv coordinates
	Vector uv;

	// flag to indicate whether this corner is contained in a face that is incident
	// to the north pole of a stereographic projection from the disk to a sphere
	bool inNorthPoleVicinity;

	// spline handle for direct editing
	KnotIter knot;

	// id between 0 and |C|-1
	int index;

private:
	// index of the halfedge associated with this corner
	int halfEdgeIndex;

	// pointer to mesh this corner belongs to
	Mesh *mesh;
};

} // namespace bff
