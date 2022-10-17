/*
The MIT License (MIT)

Copyright (c) 2017 Rohan Sawhney and Keenan Crane 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "Edge.h"
#include "Mesh.h"

namespace bff {

Edge::Edge(Mesh *mesh_):
onGenerator(false),
onCut(false),
isCuttable(true),
index(-1),
halfEdgeIndex(-1),
mesh(mesh_)
{

}

Edge::Edge(const Edge& e):
onGenerator(e.onGenerator),
onCut(e.onCut),
isCuttable(e.isCuttable),
index(e.index),
halfEdgeIndex(e.halfEdgeIndex),
mesh(e.mesh)
{

}

HalfEdgeIter Edge::halfEdge() const
{
	return mesh->halfEdges.begin() + halfEdgeIndex;
}

void Edge::setHalfEdge(HalfEdgeCIter he)
{
	halfEdgeIndex = he->index;
}

void Edge::setMesh(Mesh *mesh_)
{
	mesh = mesh_;
}

double Edge::length() const
{
	const Vector& a = halfEdge()->vertex()->position;
	const Vector& b = halfEdge()->next()->vertex()->position;

	return (b - a).norm();
}

double Edge::cotan() const
{
	return 0.5*(halfEdge()->cotan() + halfEdge()->flip()->cotan());
}

bool Edge::onBoundary() const
{
	return halfEdge()->onBoundary || halfEdge()->flip()->onBoundary;
}

} // namespace bff
