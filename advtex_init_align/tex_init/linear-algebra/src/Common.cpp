/*
The MIT License (MIT)

Copyright (c) 2017 Rohan Sawhney and Keenan Crane 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "Common.h"

namespace bff {

Common common;

Common::Common() {
	cholmod_l_start(&common);
	// common.supernodal = CHOLMOD_SUPERNODAL;

	// https://stackoverflow.com/a/22775153
	// TODO: maybe we can use CHOLMOD_AUTO instead of explicitly setting it
	common.supernodal = CHOLMOD_SIMPLICIAL;

}

Common::~Common() {
	cholmod_l_finish(&common);
}

Common::ErrorCode Common::status() const
{
	if (common.status == CHOLMOD_NOT_INSTALLED) return ErrorCode::methodNotInstalled;
	else if (common.status == CHOLMOD_OUT_OF_MEMORY) return ErrorCode::outOfMemory;
	else if (common.status == CHOLMOD_TOO_LARGE) return ErrorCode::integerOverflow;
	else if (common.status == CHOLMOD_INVALID) return ErrorCode::invalidInput;
	else if (common.status == CHOLMOD_GPU_PROBLEM) return ErrorCode::gpuProblem;
	else if (common.status == CHOLMOD_NOT_POSDEF) return ErrorCode::notPositiveDefinite;
	else if (common.status == CHOLMOD_DSMALL) return ErrorCode::smallDiagonalEntry;

	return ErrorCode::ok;
}

Common::operator cholmod_common*() {
	return &common;
}

} // namespace bff
