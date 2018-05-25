#include "../lib/classifier.hpp"
#include "../lib/image.hpp"
#include "../lib/haar_filter.hpp"

#include <vector>

#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>

Classifier::Classifier(){
	this->w1 = 1.0;
	this->w2 = 0.0;
}

Classifier::Classifier(double w1, double w2){
	this->w1 = w1;
	this->w2 = w2;
}
