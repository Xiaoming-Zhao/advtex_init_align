/*
The MIT License (MIT)

Copyright (c) 2017 Rohan Sawhney and Keenan Crane 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include "HalfEdge.h"
#include "Vertex.h"
#include "Edge.h"
#include "Face.h"
#include "Corner.h"
#include "CutIterator.h"

namespace bff {

class Mesh {
public:
	// constructor
	Mesh();

	// copy constructor
	Mesh(const Mesh& mesh);

	// returns euler characteristic
	int eulerCharacteristic() const;

	// returns mesh diameter
	double diameter() const;

	// projects uvs to PCA axis
	void projectUvsToPcaAxis();

	// range based for loop over the cut boundary
	CutPtrSet cutBoundary(); // valid only for 1 boundary loop, by default, this is the first loop

	// returns reference to wedges (a.k.a. corners)
	std::vector<Wedge>& wedges();
	const std::vector<Wedge>& wedges() const;

	// error (and warning) codes
	enum class ErrorCode {
		ok,
		nonManifoldEdges,
		nonManifoldVertices,
		isolatedVertices
	};

	// member variables
	std::vector<Vertex> vertices;
	std::vector<Edge> edges;
	std::vector<Face> faces;
	std::vector<Corner> corners;
	std::vector<HalfEdge> halfEdges;
	std::vector<Face> boundaries;
	double radius;
	Vector cm;
	ErrorCode status;
};

class Model {
public:
	// returns the number of meshes in the model
	int size() const { return (int)meshes.size(); }

	// returns the number of vertices in the model; NOTE: duplicated dofs not counted
	int nVertices() const { return (int)modelToMeshMap.size(); }

	// mesh access
	Mesh& operator[](int index) { return meshes[index]; }
	const Mesh& operator[](int index) const { return meshes[index]; }

	// vertex access
	std::pair<int, int> localVertexIndex(int index) const { return modelToMeshMap[index]; }
	int globalVertexIndex(int mesh, int index) const { return meshToModelMap[mesh][index]; }

	// clears members
	void clear() {
		meshes.clear();
		modelToMeshMap.clear();
		meshToModelMap.clear();
	}

	// members
	std::vector<Mesh> meshes;
	std::vector<std::pair<int, int>> modelToMeshMap;
	std::vector<std::vector<int>> meshToModelMap;
};

} // namespace bff
