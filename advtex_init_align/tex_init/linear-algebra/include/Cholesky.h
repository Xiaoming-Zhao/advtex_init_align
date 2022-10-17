/*
The MIT License (MIT)

Copyright (c) 2017 Rohan Sawhney and Keenan Crane 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include "Common.h"

namespace bff {

class DenseMatrix;
class SparseMatrix;

class Cholesky {
public:
	// constructor
	Cholesky(SparseMatrix& A);

	// destructor
	~Cholesky();

	// clears both symbolic and numeric factorization --
	// should be called following any change to nonzero entries
	void clear();

	// clears only numeric factorization --
	// should be called following any change to the values
	// of nonzero entries
	void clearNumeric();

	// solves positive definite
	bool solvePositiveDefinite(DenseMatrix& x, DenseMatrix& b);

protected:
	// builds symbolic factorization
	void buildSymbolic(cholmod_sparse *C);

	// builds numeric factorization
	void buildNumeric(cholmod_sparse *C);

	// updates factorizations
	void update();

	// members
	SparseMatrix& A;
	cholmod_factor *factor;
	bool validSymbolic;
	bool validNumeric;
};

} // namespace bff

#include "Cholesky.inl"
