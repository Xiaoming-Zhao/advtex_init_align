/*
The MIT License (MIT)

Copyright (c) 2017 Rohan Sawhney and Keenan Crane 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#define _USE_MATH_DEFINES

#include <stdlib.h>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include "math.h"
#include "Vector.h"

namespace bff {

class Vertex;
class Edge;
class Face;
class Corner;
class HalfEdge;
class Mesh;

typedef Corner Wedge;
typedef std::vector<Vertex>::iterator            VertexIter;
typedef std::vector<Vertex>::const_iterator      VertexCIter;
typedef std::vector<Edge>::iterator              EdgeIter;
typedef std::vector<Edge>::const_iterator        EdgeCIter;
typedef std::vector<Face>::iterator              FaceIter;
typedef std::vector<Face>::const_iterator        FaceCIter;
typedef std::vector<Corner>::iterator            CornerIter;
typedef std::vector<Corner>::const_iterator      CornerCIter;
typedef std::vector<Corner>::iterator            WedgeIter;
typedef std::vector<Corner>::const_iterator      WedgeCIter;
typedef std::vector<HalfEdge>::iterator          HalfEdgeIter;
typedef std::vector<HalfEdge>::const_iterator    HalfEdgeCIter;
typedef std::vector<Face>::iterator              BoundaryIter;
typedef std::vector<Face>::const_iterator        BoundaryCIter;

} // namespace bff
