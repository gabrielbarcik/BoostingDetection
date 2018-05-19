#ifndef FILTERH
#define FILTERH

#include <iostream>

class Filter {

public:

	Filter(int height, int width, int j_min, int i_min, int color);

	bool contains (int j, int i);
	void print_filter();

	int height;
	int width;
	int j_min;
	int i_min;
	int color; // 0 is white and 1 is black

private:
	

};

#endif