#include "../lib/filter.hpp"

Filter::Filter(int height, int width, int j_min, int i_min, int color){
	this->height = height;
	this->width = width;
	this->j_min = j_min;
	this->i_min = i_min;
	this->color = color;
}

bool Filter::contains(int j, int i){
	if ((j_min <= j && j <= j_min + width) && (i_min <= i && i <= i_min +height))
		return true;
	else
		return false;
}

void Filter::print_filter(){
	if(this->color == 0){
		std::cout << "filter characteristics: \n" << std::endl;
		std::cout << "color = white " << "height = " << this->height << " width = " << this->width <<
				" j_min = " << this->j_min << " i_min = " << i_min << std::endl;
	}

	else if (this->color == 1) {
		std::cout << "filter characteristics: \n" << std::endl;
		std::cout << "color = black " << "height = " << this->height << " width = " << this->width <<
				" j_min = " << this->j_min << " i_min = " << i_min << "\n\n" << std::endl;
	}
	
}