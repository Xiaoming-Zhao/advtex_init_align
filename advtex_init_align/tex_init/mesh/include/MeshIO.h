/*
The MIT License (MIT)

Copyright (c) 2017 Rohan Sawhney and Keenan Crane 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include <set>
#include <fstream>
#include <sstream>
#include "MeshData.h"

namespace bff {

class AdjacencyTable {
public:
	// constructs table
	void construct(int n, const std::vector<int>& indices);

	// returns unique index corresponding to entry (i, j)
	int getIndex(int i, int j) const;

	// returns table size
	int getSize() const;

private:
	// members
	std::vector<std::set<int>> data;
	std::vector<int> iMap;
	int size;
};

class PolygonSoup {
public:
	std::vector<Vector> positions;
	std::vector<int> indices;
	AdjacencyTable table; // construct after filling positions and indices
};

class MeshIO {
public:
	// reads model from obj file
	static bool read(const std::string& fileName, Model& model, std::string& error);

	// writes data to obj file
	static bool write(const std::string& fileName, Model& model,
					  const std::vector<bool>& mappedToSphere, bool normalize);
	static bool write(const std::string& fileName, Mesh& mesh);

	// separates model into components
	static void separateComponents(const PolygonSoup& soup,
								   const std::vector<int>& isCuttableEdge,
								   std::vector<PolygonSoup>& soups,
								   std::vector<std::vector<int>>& isCuttableEdgeSoups,
								   std::vector<std::pair<int, int>>& modelToMeshMap,
								   std::vector<std::vector<int>>& meshToModelMap);

	// builds a halfedge mesh
	static int buildMesh(const PolygonSoup& soup,
						  const std::vector<int>& isCuttableEdge,
						  Mesh& mesh, std::string& error);

	// centers model around origin and records radius
	static void normalize(Model& model);

private:
	// preallocates mesh elements
	static void preallocateElements(const PolygonSoup& soup, Mesh& mesh);

	// checks if mesh has isolated vertices
	static bool hasIsolatedVertices(const Mesh& mesh);

	// checks if mesh has non-manifold vertices
	static bool hasNonManifoldVertices(const Mesh& mesh);

	// reads data from obj file
	static bool read(std::istringstream& in, Model& model, std::string& error);

	// writes data to obj file
	static void write(std::ofstream& out, Model& model,
					  const std::vector<bool>& mappedToSphere, bool normalize);
	static void write(std::ofstream& out, Mesh& mesh);
};

} // namespace bff
