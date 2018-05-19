#ifndef CLASSIFIERH
#define CLASSIFIERH

#include <iostream>
#include <vector>

class Classifier {
public:

	double w1, w2;

	Classifier(void);
	std::vector<int> calculate_features(std::vector<int> features, std::vector<Classifier> classifiers);

private:

};

#endif