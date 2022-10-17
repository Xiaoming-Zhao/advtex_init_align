/*
The MIT License (MIT)

Copyright (c) 2017 Rohan Sawhney and Keenan Crane 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "HalfEdge.h"
#include "Mesh.h"

namespace bff {

HalfEdge::HalfEdge(Mesh *mesh_):
onBoundary(false),
index(-1),
nextIndex(-1),
prevIndex(-1),
flipIndex(-1),
vertexIndex(-1),
edgeIndex(-1),
faceIndex(-1),
cornerIndex(-1),
mesh(mesh_)
{

}

HalfEdge::HalfEdge(const HalfEdge& he):
onBoundary(he.onBoundary),
index(he.index),
nextIndex(he.nextIndex),
prevIndex(he.prevIndex),
flipIndex(he.flipIndex),
vertexIndex(he.vertexIndex),
edgeIndex(he.edgeIndex),
faceIndex(he.faceIndex),
cornerIndex(he.cornerIndex),
mesh(he.mesh)
{

}

HalfEdgeIter HalfEdge::next() const
{
	return mesh->halfEdges.begin() + nextIndex;
}

HalfEdgeIter HalfEdge::prev() const
{
	return mesh->halfEdges.begin() + prevIndex;
}

HalfEdgeIter HalfEdge::flip() const
{
	return mesh->halfEdges.begin() + flipIndex;
}

VertexIter HalfEdge::vertex() const
{
	return mesh->vertices.begin() + vertexIndex;
}

EdgeIter HalfEdge::edge() const
{
	return mesh->edges.begin() + edgeIndex;
}

FaceIter HalfEdge::face() const
{
	return mesh->faces.begin() + faceIndex;
}

CornerIter HalfEdge::corner() const
{
	return mesh->corners.begin() + cornerIndex;
}

WedgeIter HalfEdge::wedge() const
{
	return corner();
}

void HalfEdge::setNext(HalfEdgeCIter he)
{
	nextIndex = he->index;
}

void HalfEdge::setPrev(HalfEdgeCIter he)
{
	prevIndex = he->index;
}

void HalfEdge::setFlip(HalfEdgeCIter he)
{
	flipIndex = he->index;
}

void HalfEdge::setVertex(VertexCIter v)
{
	vertexIndex = v->index;
}

void HalfEdge::setEdge(EdgeCIter e)
{
	edgeIndex = e->index;
}

void HalfEdge::setFace(FaceCIter f)
{
	faceIndex = f->index;
}

void HalfEdge::setCorner(CornerCIter c)
{
	cornerIndex = c->index;
}

void HalfEdge::setMesh(Mesh *mesh_)
{
	mesh = mesh_;
}

double HalfEdge::cotan() const
{
	if (onBoundary) return 0.0;

	const Vector& a = vertex()->position;
	const Vector& b = next()->vertex()->position;
	const Vector& c = prev()->vertex()->position;

	Vector u = a - c;
	Vector v = b - c;

	double w = dot(u, v)/cross(u, v).norm();
	if (std::isinf(w) || std::isnan(w)) w = 0.0;
	return w;
}

} // namespace bff
