/*
The MIT License (MIT)

Copyright (c) 2017 Rohan Sawhney and Keenan Crane 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include <vector>

namespace bff {

class DisjointSets {
public:
	// constructor
	DisjointSets(int n_): parent(n_ + 1), rank(n_ + 1), marked(n_ + 1) {
		// initialize all elements to be in different sets and to have rank 0
		for (int i = 0; i <= n_; i++) {
			rank[i] = 0;
			parent[i] = i;
			marked[i] = false;
		}
	}

	// find parent of element x
	int find(int x) {
		if (x != parent[x]) parent[x] = find(parent[x]);
		return parent[x];
	}

	// union by rank
	// if either set is marked, then the result is marked
	void merge(int x, int y) {
		x = find(x);
		y = find(y);

		// smaller tree becomes a subtree of the larger tree
		if (rank[x] > rank[y]) parent[y] = x;
		else parent[x] = y;

		if (rank[x] == rank[y]) rank[y]++;

		// if either set was marked, both are marked
		if (marked[x] || marked[y]) {
			marked[x] = true;
			marked[y] = true;
		}
	}

	// mark set
	void mark(int x) {
		marked[find(x)] = true;
	}

	// unmark set
	void unmark(int x) {
		marked[find(x)] = false;
	}

	// check if set is marked
	bool isMarked(int x) {
		return marked[find(x)];
	}

private:
	// members
	std::vector<int> parent;
	std::vector<int> rank;
	std::vector<bool> marked;
};

} // namespace bff
