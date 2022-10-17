/*
The MIT License (MIT)

Copyright (c) 2017 Rohan Sawhney and Keenan Crane 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include "math.h"

namespace bff {

class Vector {
public:
	// initializes all components to zero
	Vector();

	// initializes with specified components
	Vector(double x, double y, double z = 0.0);

	// copy constructor
	Vector(const Vector& v);

	// access
	double& operator[](int index);
	const double& operator[](int index) const;

	// math
	Vector operator*(double s) const;
	Vector operator/(double s) const;
	Vector operator+(const Vector& v) const;
	Vector operator-(const Vector& v) const;
	Vector operator-() const;

	Vector& operator*=(double s);
	Vector& operator/=(double s);
	Vector& operator+=(const Vector& v);
	Vector& operator-=(const Vector& v);

	// returns Euclidean length
	double norm() const;

	// returns Euclidean length squared
	double norm2() const;

	// normalizes vector
	void normalize();

	// returns unit vector in the direction of this vector
	Vector unit() const;

	// members
	double x, y, z;
};

Vector operator*(double s, const Vector& v);
double dot(const Vector& u, const Vector& v);
Vector cross(const Vector& u, const Vector& v);

} // namespace bff

#include "Vector.inl"
