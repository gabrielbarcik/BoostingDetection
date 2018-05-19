#include "../lib/classifier.hpp"
#include <vector>

Classifier::Classifier(){
	this->w1 = 1.0;
	this->w2 = 0.0;
}

std::vector<int> Classifier::calculate_features(std::vector<int> features, std::vector<Classifier> classifiers){
	std::vector<int> h;
	int K = 10000; // TODO: evaluate the right value for K
	double epsilon = 0.0005; // TODO: evaluate the right value for epsilon

	for(int k = 0; k < K; k++){
		for(unsigned long i = 0; i < features.size(); i++){
			//Image imgage = find_random_apprentissage_image(); // TODO: create function that choose randomly 
															  // an imagem from the base d’apprentissage

			//int Xki = find_i_componente(image); // TODO: calculate the i component of the features vector of image
			int ck; // TODO: o que é ck?

			//classifiers[]
		}

	}

	return h;
}