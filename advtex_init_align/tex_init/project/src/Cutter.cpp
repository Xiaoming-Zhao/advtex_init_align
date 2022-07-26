/*
The MIT License (MIT)

Copyright (c) 2017 Rohan Sawhney and Keenan Crane 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "Cutter.h"
#include "DisjointSets.h"
#include <queue>
#include <tuple>
#include <functional>

namespace bff {

typedef std::tuple<double, VertexIter, HalfEdgeIter> VertexEntry;

void Cutter::cut(const std::vector<VertexIter>& cones, Mesh& mesh)
{
	// initialize data structures
	int nV = (int)mesh.vertices.size();
	DisjointSets ds(nV);
	VertexData<HalfEdgeIter> parent(mesh, mesh.halfEdges.end());
	std::priority_queue<VertexEntry, std::vector<VertexEntry>, std::greater<VertexEntry>> pq;

	// merge each boundary loop
	for (BoundaryCIter b = mesh.boundaries.begin(); b != mesh.boundaries.end(); b++) {
		bool first = true;
		HalfEdgeIter he = b->halfEdge();
		do {
			if (first) first = false;
			else ds.merge(he->vertex()->index, he->next()->vertex()->index);

			// add initial entries to pq
			HalfEdgeIter vHe = he;
			do {
				EdgeCIter e = vHe->edge();
				if (!e->onBoundary() && e->isCuttable) {
					pq.push(VertexEntry(e->length(), vHe->flip()->vertex(), vHe));
				}

				vHe = vHe->flip()->next();
			} while (vHe != he);

			he = he->next();
		} while (he != b->halfEdge());

		// mark set
		ds.mark(he->vertex()->index);
	}

	// add cone neighbors to pq
	for (int i = 0; i < (int)cones.size(); i++) {
		HalfEdgeIter he = cones[i]->halfEdge();
		do {
			if (he->edge()->isCuttable) {
				pq.push(VertexEntry(he->edge()->length(), he->flip()->vertex(), he));
			}

			he = he->flip()->next();
		} while (he != cones[i]->halfEdge());

		// mark set
		ds.mark(cones[i]->index);
	}

	// construct approximate steiner tree
	while (!pq.empty()) {
		VertexEntry entry = pq.top();
		pq.pop();

		double weight = std::get<0>(entry);
		HalfEdgeIter he = std::get<2>(entry);
		VertexIter v1 = std::get<1>(entry);
		VertexIter v2 = he->vertex();

		if (ds.find(v1->index) != ds.find(v2->index)) {
			// if merging two marked sets, then mark edges that connect these
			// two regions as cut edges
			if (ds.isMarked(v1->index) && ds.isMarked(v2->index)) {
				// one side
				HalfEdgeIter currHe = parent[v1];
				while (currHe != mesh.halfEdges.end()) {
					currHe->edge()->onCut = true;
					currHe = parent[currHe->vertex()];
				}

				// bridge
				he->edge()->onCut = true;

				// other side
				currHe = parent[v2];
				while (currHe != mesh.halfEdges.end()) {
					currHe->edge()->onCut = true;
					currHe = parent[currHe->vertex()];
				}
			}

			// record potential parent and merge sets
			parent[v1] = he;
			ds.merge(v1->index, v2->index);

			// add neighbors
			HalfEdgeIter vHe = v1->halfEdge();
			do {
				if (vHe->edge()->isCuttable) {
					pq.push(VertexEntry(vHe->edge()->length() + weight,
										vHe->flip()->vertex(), vHe));
				}

				vHe = vHe->flip()->next();
			} while (vHe != v1->halfEdge());
		}
	}
}

void Cutter::glue(Mesh& mesh)
{
	for (EdgeIter e = mesh.edges.begin(); e != mesh.edges.end(); e++) {
		e->onCut = false;
	}
}

} // namespace bff
