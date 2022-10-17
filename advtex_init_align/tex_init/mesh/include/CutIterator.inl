/*
The MIT License (MIT)

Copyright (c) 2017 Rohan Sawhney and Keenan Crane 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "Edge.h"

namespace bff {

inline CutPtrIterator::CutPtrIterator(HalfEdgeIter he, bool justStarted_):
currHe(he),
justStarted(justStarted_)
{

}

inline const CutPtrIterator& CutPtrIterator::operator++()
{
	justStarted = false;
	HalfEdgeIter h = currHe->flip();
	do {

		h = h->prev()->flip(); // loop around one ring counter clockwise
	} while (!h->onBoundary && !h->edge()->onCut);

	currHe = h;
	return *this;
}

inline bool CutPtrIterator::operator==(const CutPtrIterator& other) const
{
	return currHe == other.currHe && justStarted == other.justStarted;
}

inline bool CutPtrIterator::operator!=(const CutPtrIterator& other) const
{
	return !(*this == other);
}

inline WedgeIter CutPtrIterator::operator*() const
{
	return currHe->flip()->prev()->corner();
}

inline CutPtrSet::CutPtrSet():
isInvalid(true)
{

}

inline CutPtrSet::CutPtrSet(HalfEdgeIter he):
firstHe(he),
isInvalid(false)
{

}

inline CutPtrIterator CutPtrSet::begin()
{
	if (isInvalid) return CutPtrIterator(firstHe, false);

	return CutPtrIterator(firstHe, true);
}

inline CutPtrIterator CutPtrSet::end()
{
	return CutPtrIterator(firstHe, false);
}

} // namespace bff
