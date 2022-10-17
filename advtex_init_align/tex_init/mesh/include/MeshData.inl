/*
The MIT License (MIT)

Copyright (c) 2017 Rohan Sawhney and Keenan Crane 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

namespace bff {

template<typename Element, std::vector<Element> Mesh::*meshList, typename ElementIter, typename T>
MeshData<Element, meshList, ElementIter, T>::MeshData(const Mesh& mesh_):
data((mesh_.*meshList).size())
{

}

template<typename Element, std::vector<Element> Mesh::*meshList, typename ElementIter, typename T>
MeshData<Element, meshList, ElementIter, T>::MeshData(const Mesh& mesh_, const T& initVal):
data((mesh_.*meshList).size(), initVal)
{

}

template<typename Element, std::vector<Element> Mesh::*meshList, typename ElementIter, typename T>
T& MeshData<Element, meshList, ElementIter, T>::operator[](ElementIter e)
{
	return data[e->index];
}

template<typename Element, std::vector<Element> Mesh::*meshList, typename ElementIter, typename T>
const T& MeshData<Element, meshList, ElementIter, T>::operator[](ElementIter e) const
{
	return data[e->index];
}

template<typename T>
using VertexData = MeshData<Vertex, &Mesh::vertices, VertexCIter, T>;

template<typename T>
using EdgeData = MeshData<Edge, &Mesh::edges, EdgeCIter, T>;

template<typename T>
using FaceData = MeshData<Face, &Mesh::faces, FaceCIter, T>;

template<typename T>
using BoundaryData = MeshData<Face, &Mesh::boundaries, BoundaryCIter, T>;

template<typename T>
using HalfEdgeData = MeshData<HalfEdge, &Mesh::halfEdges, HalfEdgeCIter, T>;

template<typename T>
using CornerData = MeshData<Corner, &Mesh::corners, CornerCIter, T>;

template<typename T>
using WedgeData = MeshData<Wedge, &Mesh::corners, WedgeCIter, T>;

} // namespace bff
