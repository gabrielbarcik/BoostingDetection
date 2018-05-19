#ifndef HAAR_FILTERH
#define HAAR_FILTERH

#include "filter.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

class Haar_filter {


public:

	Haar_filter(int type, int height, int width, int j_min, int i_min);

	bool contains (int j, int i); 
	int feature(cv::Mat &integral_image); // TODO: create the function to calculate the feature

	int type;
	/*  ----> j  (width)
		|
(height)|
		v
		i

		Type 1
		-------------
		|00000|11111|
		|00000|11111|
		|00000|11111|
		|00000|11111|
		-------------

		Type 2
		-------
		|11111|
		|11111|
		-------
		|00000|
		|00000|
		-------

		Type 3
		-------------------
		|00000|11111|00000|
		|00000|11111|00000|
		|00000|11111|00000|
		|00000|11111|00000|
		-------------------

		Type 4
		-------------
		|00000|11111|
		|00000|11111|
		|11111|00000|
		|11111|00000|
		-------------

	*/
	
	std::vector<Filter> white_filters;
	std::vector<Filter> black_filters;
private:
	

};

#endif