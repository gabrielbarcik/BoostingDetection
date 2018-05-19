#include "../lib/haar_filter.hpp"
#include <iostream>
#include <vector>

Haar_filter::Haar_filter(int type, int height, int width, int j_min, int i_min){
	this->type = type;

	if (type == 1){
		/*  type 1
		-------------
		|00000|11111|
		|00000|11111|
		|00000|11111|
		|00000|11111|
		-------------
		*/
		Filter white = Filter(height, width/2, j_min, i_min, 0);
		Filter black = Filter(height, width/2, j_min + width/2, i_min, 1);

		this->white_filters.push_back(white);
		this->black_filters.push_back(black);

	} else if (type == 2){
		/*	type 2
			-------
			|11111|
			|11111|
			-------
			|00000|
			|00000|
			-------
		*/

		Filter white = Filter(height/2, width, j_min, i_min, 0);
		Filter black = Filter(height/2, width, j_min, i_min + height/2, 1);

		this->white_filters.push_back(white);
		this->black_filters.push_back(black);

	} else if (type == 3) {
		/*  type 3
			-------------------
			|00000|11111|00000|
			|00000|11111|00000|
			|00000|11111|00000|
			|00000|11111|00000|
			-------------------
		*/

		Filter white1 = Filter(height, width/3			   , j_min, i_min, 0);
		Filter black  = Filter(height, width/3  + width % 3, j_min + width/3, i_min, 1);
		Filter white2 = Filter(height, width/3			   , j_min + 2*(width/3)  + width % 3, i_min, 0);

		this->white_filters.push_back(white1);
		this->white_filters.push_back(white2);
		this->black_filters.push_back(black);

	} else if (type == 4) {
		/*	type 4
			-------------
			|00000|11111|
			|00000|11111|
			|-----|-----|
			|11111|00000|
			|11111|00000|
			-------------
		*/

		Filter white1 = Filter(height/2				, width/2			 , j_min		  , i_min, 0);
		Filter black1 = Filter(height/2				, width/2 + width % 2, j_min + width/2, i_min, 1);
		Filter black2 = Filter(height/2 + height % 2, width/2 			 , j_min		  , i_min + height/2, 1);
		Filter white2 = Filter(height/2 + height % 2, width/2 + width % 2, j_min + width/2, i_min + height/2, 0);
		

		this->white_filters.push_back(white1);
		this->white_filters.push_back(white2);
		this->black_filters.push_back(black1);
		this->black_filters.push_back(black2);

	} else {
		// TODO: error, it is not a valide type
	}
}

bool Haar_filter::contains(int j, int i){
	for(int k = 0; k < this->black_filters.size(); k++){
		if(this->black_filters[k].contains(j,i))
			return true;
	}

	for(int k = 0; k < this->white_filters.size(); k++){
		if(this->white_filters[k].contains(j,i))
			return true;
	}

	return false;
}

int Haar_filter::feature(cv::Mat &integral_image){
	int feature = 0;
	Filter f = Filter(0,0,0,0,0); // just for initialization

	for(int k = 0; k < this->black_filters.size(); k++){
		f = this->black_filters[k];
		feature += integral_image.at<uchar>(f.i_min + f.height,f.j_min + f.width) 
					- integral_image.at<uchar>(f.i_min + f.height,f.j_min)
					- integral_image.at<uchar>(f.i_min,f.j_min + f.width)
					+ integral_image.at<uchar>(f.i_min,f.j_min);

	}

	for(int k = 0; k < this->white_filters.size(); k++){
		f = this->white_filters[k];
		feature -= integral_image.at<uchar>(f.i_min + f.height,f.j_min + f.width) 
					- integral_image.at<uchar>(f.i_min + f.height,f.j_min)
					- integral_image.at<uchar>(f.i_min,f.j_min + f.width)
					+ integral_image.at<uchar>(f.i_min,f.j_min);
	}

	return feature;
}
