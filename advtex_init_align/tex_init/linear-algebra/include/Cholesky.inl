/*
The MIT License (MIT)

Copyright (c) 2017 Rohan Sawhney and Keenan Crane 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "SparseMatrix.h"

namespace bff {

inline Cholesky::Cholesky(SparseMatrix& A_):
A(A_),
factor(NULL),
validSymbolic(false),
validNumeric(false)
{

}

inline Cholesky::~Cholesky()
{
	clear();
}

inline void Cholesky::clear()
{
	if (factor) {
		cholmod_l_free_factor(&factor, common);
		factor = NULL;
	}

	validSymbolic = false;
	validNumeric = false;
}

inline void Cholesky::clearNumeric()
{
	validNumeric = false;
}

inline void Cholesky::buildSymbolic(cholmod_sparse *C)
{
	clear();

	factor = cholmod_l_analyze(C, common);
	if (factor) validSymbolic = true;
}

inline void Cholesky::buildNumeric(cholmod_sparse *C)
{
	if (factor) validNumeric = (bool)cholmod_l_factorize(C, factor, common);
}

inline void Cholesky::update()
{
	cholmod_sparse *C = A.toCholmod();
	C->stype = 1;

	if (!validSymbolic) buildSymbolic(C);
	if (!validNumeric) buildNumeric(C);
}

inline bool Cholesky::solvePositiveDefinite(DenseMatrix& x, DenseMatrix& b)
{
	update();
	if (factor) x = cholmod_l_solve(CHOLMOD_A, factor, b.toCholmod(), common);

	return common.status() == Common::ErrorCode::ok;
}

} // namespace bff
