/*
The MIT License (MIT)

Copyright (c) 2017 Rohan Sawhney and Keenan Crane 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include <map>
#include "math.h"

typedef std::map<double, double>::iterator       KnotIter;
typedef std::map<double, double>::const_iterator KnotCIter;

class Spline {
public:
	// returns the interpolated value
	double evaluate(double t) const;

	// sets the value of the spline at a given time (i.e., knot),
	// creating a new knot at this time if necessary
	KnotIter addKnot(double t, double p);

	// removes the knot closest to the given time, within the given tolerance
	// returns true iff a knot was removed
	bool removeKnot(double t, double tolerance = 0.001);

	// sets knot entries to zero
	void reset();

	// members
	std::map<double, double> knots;
};

#include "Spline.inl"
