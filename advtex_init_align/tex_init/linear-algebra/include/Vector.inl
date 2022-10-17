/*
The MIT License (MIT)

Copyright (c) 2017 Rohan Sawhney and Keenan Crane 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

namespace bff {

inline Vector::Vector():
x(0.0),
y(0.0),
z(0.0)
{

}

inline Vector::Vector(double x_, double y_, double z_):
x(x_),
y(y_),
z(z_)
{

}

inline Vector::Vector(const Vector& v):
x(v.x),
y(v.y),
z(v.z)
{

}

inline double& Vector::operator[](int index)
{
	return (&x)[index];
}

inline const double& Vector::operator[](int index) const
{
	return (&x)[index];
}

inline Vector Vector::operator*(double s) const
{
	return Vector(x*s, y*s, z*s);
}

inline Vector Vector::operator/(double s) const
{
	return (*this)*(1.0/s);
}

inline Vector Vector::operator+(const Vector& v) const
{
	return Vector(x + v.x, y + v.y, z + v.z);
}

inline Vector Vector::operator-(const Vector& v) const
{
	return Vector(x - v.x, y - v.y, z - v.z);
}

inline Vector Vector::operator-() const
{
	return Vector(-x, -y, -z);
}

inline Vector& Vector::operator*=(double s)
{
	x *= s;
	y *= s;
	z *= s;

	return *this;
}

inline Vector& Vector::operator/=(double s)
{
	(*this) *= (1.0/s);

	return *this;
}

inline Vector& Vector::operator+=(const Vector& v)
{
	x += v.x;
	y += v.y;
	z += v.z;

	return *this;
}

inline Vector& Vector::operator-=(const Vector& v)
{
	x -= v.x;
	y -= v.y;
	z -= v.z;

	return *this;
}

inline double Vector::norm() const
{
	return sqrt(norm2());
}

inline double Vector::norm2() const
{
	return dot(*this, *this);
}

inline void Vector::normalize()
{
	(*this) /= norm();
}

inline Vector Vector::unit() const
{
   return (*this) / norm();
}

inline Vector operator*(double s, const Vector& v)
{
	return v*s;
}

inline double dot(const Vector& u, const Vector& v)
{
	return u.x*v.x + u.y*v.y + u.z*v.z;
}

inline Vector cross(const Vector& u, const Vector& v)
{
	return Vector(u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x);
}

} // namespace bff
