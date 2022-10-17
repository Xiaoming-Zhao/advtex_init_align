/*
The MIT License (MIT)

Copyright (c) 2017 Rohan Sawhney and Keenan Crane 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "Vertex.h"
#include "Mesh.h"

namespace bff {

Vertex::Vertex(Mesh *mesh_):
inNorthPoleVicinity(false),
index(-1),
referenceIndex(-1),
halfEdgeIndex(-1),
mesh(mesh_)
{

}

Vertex::Vertex(const Vertex& v):
position(v.position),
inNorthPoleVicinity(v.inNorthPoleVicinity),
index(v.index),
referenceIndex(v.referenceIndex),
halfEdgeIndex(v.halfEdgeIndex),
mesh(v.mesh)
{

}

HalfEdgeIter Vertex::halfEdge() const
{
	return mesh->halfEdges.begin() + halfEdgeIndex;
}

void Vertex::setHalfEdge(HalfEdgeCIter he)
{
	halfEdgeIndex = he->index;
}

WedgeIter Vertex::wedge() const
{
	HalfEdgeCIter h = halfEdge();
	while (h->onBoundary || !h->next()->wedge()->isReal()) {
		h = h->flip()->next();
	}

	return h->next()->wedge();
}

void Vertex::setMesh(Mesh *mesh_)
{
	mesh = mesh_;
}

bool Vertex::onBoundary(bool checkIfOnCut) const
{
	if (inNorthPoleVicinity) return true;

	HalfEdgeCIter h = halfEdge();
	do {
		if (h->onBoundary) return true;
		if (checkIfOnCut && h->edge()->onCut) return true;

		h = h->flip()->next();
	} while (h != halfEdge());

	return false;
}

bool Vertex::isIsolated() const
{
	return halfEdgeIndex == -1;
}

int Vertex::degree() const
{
	int k = 0;
	HalfEdgeCIter h = halfEdge();
	do {
		k++;

		h = h->flip()->next();
	} while (h != halfEdge());

	return k;
}

int Vertex::degreeDebug() const
{
	int k = 0;
	HalfEdgeCIter h = halfEdge();
	std::cout << h->vertex()->index << std::endl;
	do {
		k++;

		h = h->flip()->next();
	} while (h != halfEdge());

	return k;
}

Vector Vertex::normal() const
{
	Vector n;
	HalfEdgeCIter h = halfEdge();
	do {
		if (!h->onBoundary) n += h->face()->normal(false)*h->next()->corner()->angle();

		h = h->flip()->next();
	} while (h != halfEdge());

	n.normalize();

	return n;
}

double Vertex::angleDefect() const
{
	double sum = 0.0;
	if (onBoundary()) return sum;

	HalfEdgeCIter h = halfEdge();
	do {
		sum += h->next()->corner()->angle();

		h = h->flip()->next();
	} while (h != halfEdge());

	return 2*M_PI - sum;
}

double Vertex::exteriorAngle() const
{
	double sum = 0.0;
	if (!onBoundary()) return sum;

	HalfEdgeCIter h = halfEdge();
	do {
		if (!h->onBoundary) sum += h->next()->corner()->angle();

		h = h->flip()->next();
	} while (h != halfEdge());

	return M_PI - sum;
}

} // namespace bff
