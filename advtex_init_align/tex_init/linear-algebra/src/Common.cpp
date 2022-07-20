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
