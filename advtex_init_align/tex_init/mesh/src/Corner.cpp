/*
The MIT License (MIT)

Copyright (c) 2017 Rohan Sawhney and Keenan Crane 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "Corner.h"
#include "Mesh.h"

namespace bff {

Corner::Corner(Mesh *mesh_):
inNorthPoleVicinity(false),
index(-1),
halfEdgeIndex(-1),
mesh(mesh_)
{

}

Corner::Corner(const Corner& c):
uv(c.uv),
inNorthPoleVicinity(c.inNorthPoleVicinity),
knot(c.knot),
index(c.index),
halfEdgeIndex(c.halfEdgeIndex),
mesh(c.mesh)
{

}

HalfEdgeIter Corner::halfEdge() const
{
	return mesh->halfEdges.begin() + halfEdgeIndex;
}

void Corner::setHalfEdge(HalfEdgeCIter he)
{
	halfEdgeIndex = he->index;
}

VertexIter Corner::vertex() const
{
	return halfEdge()->prev()->vertex();
}

FaceIter Corner::face() const
{
	return halfEdge()->face();
}

CornerIter Corner::next() const
{
	return halfEdge()->next()->corner();
}

CornerIter Corner::prev() const
{
	return halfEdge()->prev()->corner();
}

WedgeIter Wedge::nextWedge() const
{
	bool noCut = true;
	HalfEdgeCIter h = halfEdge()->prev();
	do {
		if (h->edge()->onCut) {
			noCut = false;
			break;
		}

		h = h->flip()->next();
	} while (!h->onBoundary);

	return noCut ? h->prev()->flip()->prev()->corner() :
				   h->prev()->corner();
}

void Corner::setMesh(Mesh *mesh_)
{
	mesh = mesh_;
}

double Corner::angle() const
{
	const Vector& a = halfEdge()->prev()->vertex()->position;
	const Vector& b = halfEdge()->vertex()->position;
	const Vector& c = halfEdge()->next()->vertex()->position;

	Vector u = b - a;
	u.normalize();

	Vector v = c - a;
	v.normalize();

	return acos(std::max(-1.0, std::min(1.0, dot(u, v))));
}

bool Corner::isReal() const
{
	return !inNorthPoleVicinity;
}

double Wedge::exteriorAngle() const
{
	HalfEdgeCIter h = halfEdge()->prev();

	double sum = 0.0;
	if (!h->vertex()->onBoundary()) return sum;

	do {
		sum += h->next()->corner()->angle();
		if (h->edge()->onCut) break;

		h = h->flip()->next();
	} while (!h->onBoundary);

	return M_PI - sum;
}

double Wedge::scaling() const
{
	WedgeCIter next = nextWedge();

	Vector a = prev()->uv;
	Vector b = uv;
	Vector c = next->uv;

	double lij = (b - a).norm();
	double ljk = (c - b).norm();

	double uij = log(lij/halfEdge()->next()->edge()->length());
	double ujk = log(ljk/next->halfEdge()->next()->edge()->length());

	return (lij*uij + ljk*ujk)/(lij + ljk);
}

Vector Wedge::tangent() const
{
	Vector a = prev()->uv;
	Vector b = uv;
	Vector c = nextWedge()->uv;

	Vector Tij = b - a;
	Tij.normalize();

	Vector Tjk = c - b;
	Tjk.normalize();

	Vector T = Tij + Tjk;
	T.normalize();

	return T;
}

} // namespace bff
