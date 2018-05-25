#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../lib/image.hpp"
#include "../lib/haar_filter.hpp"
#include "../lib/classifier.hpp"
#include <chrono>
#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>

#include <string>
#include <math.h>
#include <float.h>

#include <mpi.h> 

void print (int matrix[3][3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void matrix_integral () {
    int matrix[3][3];
    int integral[3][3];
    int cumulative_row = 0;

    // matrix creation
    int counter = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            matrix[i][j] = ++counter;
        }
    }

    // first element -> trivial initialization
    integral[0][0] = matrix[0][0];
    // first line -> image[-1][y] = 0 for all y
    cumulative_row = matrix[0][0];
    for (int j = 1; j < 3; j++) {
        integral[0][j] = cumulative_row + matrix[0][j];
        cumulative_row += matrix[0][j];
    }

    for (int i = 1; i < 3; i++) {
        cumulative_row = 0;
        for (int j = 0; j < 3; j++) {
            cumulative_row += matrix[i][j];
            integral[i][j] = integral[i-1][j] + cumulative_row;
        }
    }

    print(matrix);
    print(integral);
}

void integral_transform (cv::Mat &image, cv::Mat &integral_image) {

    int cumulative_row = 0;

    // first element, trivial initialization
    integral_image.at<uchar>(0, 0) = image.at<uchar>(0, 0);

    // first line -> image[-1][y] = 0 for all y
    cumulative_row = image.at<uchar>(0, 0);
    for (int j = 1; j < image.cols; j++) {
        integral_image.at<uchar>(0, j) = cumulative_row + image.at<uchar>(0, j);
        cumulative_row += image.at<uchar>(0, j);
    }

    // all the rest of the matrix
    for (int i = 1; i < image.rows; i++) {
        cumulative_row = 0;
        for (int j = 0; j < image.cols; j++) {
            cumulative_row += image.at<uchar>(i, j);
            integral_image.at<uchar>(i, j) = integral_image.at<uchar>(i-1, j) + cumulative_row;
        }
    }
}

void create_filters (std::vector<Haar_filter> &filters) {
    // parameters
    int const width = 8;
    int const height = 8;
    int const max_width = 112;
    int const max_height = 92;

    // for all sizes of filters
    // for all positions in the image

    for (int h = height; h <= max_height; h += 4) {
        for (int w = width; w <= max_width; w += 4) {
            for (int i = 0; i < max_height; i++) {
                for (int j = 0; j < max_width; j++) {
                    // filter out of the image
                    if (j + w > max_width - j || i + h > max_height - i)
                        continue;
                    for (int type = 1; type <= 4; type++)
                        filters.push_back(Haar_filter(type, h, w, j, i));
                }

            }

        }
    }

}

// calculate the number of files of a folder with path folder_name
int number_of_files(std::string folder_name) {
    DIR *dp;
    int num_files = -2; // starts at -2 because readdir takes "." and ".." as a file.
    struct dirent *ep;

    const char * c = folder_name.c_str();
    dp = opendir (c);

    if (dp != NULL) {
        while (ep = readdir (dp))
          num_files++;

        (void) closedir (dp);
    } else
        perror ("Couldn't open the directory");

    return num_files;
}


/* funcion used in calculate_features to find randomly a image in the learning base and calculate
    it`s features and modify ck. ck = 1 if the image has a visage and -1 if not.
*/
void find_random_image_learning (int NUMBER_FILES_POS, int NUMBER_FILES_NEG, int &ck, std::vector<int> &features, std::vector<Haar_filter> &filters, std::vector<Classifier> &classifiers){
    double r = ((double) rand() / (RAND_MAX));
    int r_int = 1;
    std::string image_path;

    if(r < 0.5){
        ck = -1;
        image_path = "neg/";
        srand((unsigned)time(NULL));
        r = (rand()%(NUMBER_FILES_NEG)) + 1.0;
        r_int = (int) r;

        image_path = "neg/im" + std::to_string(r_int) + ".jpg";
        // std::cout << image_path << std::endl;
        
    } else {
        ck = 1;
        image_path = "pos/";
        srand((unsigned)time(NULL));
        r = (rand()%(NUMBER_FILES_POS)) + 1.0;
        r_int = (int) r;

        image_path = "pos/im" + std::to_string(r_int) + ".jpg";
        // std::cout << image_path << std::endl;
    }

    cv::Mat image = cv::imread(image_path);
    Image img = Image(image_path, image);

    auto start_feature = std::chrono::high_resolution_clock::now();
    for (unsigned long i = 0; i < filters.size(); i++) {
        features.push_back(filters[i].feature(img.integral_image));
    }
    auto finish_feature = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_feature = finish_feature - start_feature;
    std::cout << "Elapsed Time (Calculate Feature): " << elapsed_feature.count() << std::endl;
}

int classifier_h_of_a_feature (Classifier classifier, double feature_of_image) {
	if (classifier.w1 * feature_of_image + classifier.w2 >= 0.0)
		return 1;
	else 
		return -1;
}

void train_model(std::vector<Classifier> &classifiers, std::vector<Haar_filter> &filters){
    int K = 4; // TODO: evaluate the right value for K
    double epsilon = 0.01; // TODO: evaluate the right value for epsilon
    std::vector<int> features;
    long Xki, h;
    int NUMBER_FILES_POS = number_of_files("pos/");
    int NUMBER_FILES_NEG = number_of_files("neg/");

    for(int k = 1, ck; k <= K; k++){
        auto start_random = std::chrono::high_resolution_clock::now();
        find_random_image_learning(NUMBER_FILES_POS, NUMBER_FILES_NEG, ck, features, filters, classifiers); 
        auto finish_random = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_random = finish_random - start_random;

        auto start_classify = std::chrono::high_resolution_clock::now();
        for(unsigned long i = 0; i < features.size(); i++){
            Xki = features[i];

            h = classifier_h_of_a_feature(classifiers[i], features[i]);
            classifiers[i].w1 -= epsilon * (h - ck) * Xki;
            classifiers[i].w2 -= epsilon * (h - ck);
        }
        auto finish_classify = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_classify = finish_classify - start_classify;
        features.clear();
    }
}

std::vector<Classifier> initialize_classifier_vector(unsigned long size){
	std::vector<Classifier> vec;

	for(unsigned long i = 0; i < size; i++)
		vec.push_back(Classifier());

	return vec;
}

std::vector<double> initialize_weights(std::vector<double> &weights_lambda, int n){
	double base_case = 1 / (double) n;
	for(unsigned long i = 0; i < n; i++){
		weights_lambda.push_back(base_case);
	}
}

int function_of_error_E (int h, int c){
	if (h == c){
		return 0;
	} else
		return 1;
}


int classifier_h_of_a_image(std::vector<int> features_of_image, std::vector<Classifier> classifiers){
	int num_positives = 0;

	for(unsigned long i = 0; i < features_of_image.size(); i++){
			if (classifier_h_of_a_feature(classifiers[i], features_of_image[i]) == 1)
				num_positives++;
	}

	if(num_positives > features_of_image.size()/2)
		return 1;
	else
		return -1;
}

void calculate_features_of_all_images(std::vector< std::vector<int> > &features_of_images,
		std::vector<int> &c, std::vector<Haar_filter> &filters, int NUMBER_FILES_POS, int NUMBER_FILES_NEG){
	
    std::string image_path;
    cv::Mat image;

	for(int i = 1; i <= NUMBER_FILES_NEG; i++){
        image_path = "neg/im" + std::to_string(i) + ".jpg";
        image = cv::imread(image_path);
    	Image img = Image(image_path, image);
    	    for (unsigned long j = 0; j < filters.size(); j++) {
        		features_of_images[i].push_back(filters[j].feature(img.integral_image));
        		c.push_back(-1);
    		}
	}
       
    for(int i = NUMBER_FILES_NEG + 1; i <= NUMBER_FILES_POS + NUMBER_FILES_NEG; i++){
        image_path = "pos/im" + std::to_string(i) + ".jpg";
        image = cv::imread(image_path);
    	 Image img = Image(image_path, image);
    	    for (unsigned long j = 0; j < filters.size(); j++) {
        		features_of_images[i].push_back(filters[j].feature(img.integral_image));
        		c.push_back(1);
    		}
	}

}

void update_weight(std::vector<double> &weights_lambda, Classifier best_h, 
					std::vector<Classifier> &classifiers, std::vector< std::vector<int> > features_of_images,
					int i_minimisator, std::vector<int> c, double alfa) {
	double sum = 0.0;

	// update weights
    for(int i = 0; i < weights_lambda.size(); i++){
    	weights_lambda[i] *= exp(-c[i]*alfa*classifier_h_of_a_feature(best_h, features_of_images[i_minimisator][i]));
    	sum += weights_lambda[i];
    }

    // normalization
    for(int i = 0; i < weights_lambda.size(); i++){
    	weights_lambda[i] /= sum;
    }
}

void initialize_F(std::vector<Classifier> &classifiers_result_F, int size){
	for (int i = 0; i < size; i++){
		classifiers_result_F.push_back(Classifier(0.0,0.0));
	}
}

void boosting_classifiers(std::vector<Classifier> &classifiers, std::vector<Haar_filter> &filters,
									std::vector<Classifier> &final_classifier){
	int NUMBER_FILES_POS = number_of_files("dev/neg");
    int NUMBER_FILES_NEG = number_of_files("dev/pos");

    int n = NUMBER_FILES_POS + NUMBER_FILES_NEG;


    /*std::vector<double> weights_lambda;
    initialize_weights(weights_lambda, n);

    initialize_F(final_classifier, classifiers.size());
    double teta = 0.789; // TODO: evaluete a good number for teta
    int num_interaction_N = 1000; // TODO: evaluete a good number for N

	std::vector< std::vector<int> > features_of_images;
	std::vector<int> c;
	calculate_features_of_all_images(features_of_images, c, filters, NUMBER_FILES_POS, NUMBER_FILES_NEG);

	// find best h_i
	double epsilon_i, epsilon_min = DBL_MAX, alfa;
	Classifier best_h;
	int i_minimisator;

	for(int k = 0; k < num_interaction_N; k++) {

		epsilon_min = DBL_MAX;
		for(int i = 0; i < features_of_images.size(); i++) { 
			epsilon_i = 0.0;
			for(int j = 0; j < n; j++){
				epsilon_i += weights_lambda[j]*function_of_error_E(
					classifier_h_of_a_feature(classifiers[i], features_of_images[j][i]),
					c[j]);
			}

			if(epsilon_i < epsilon_min){
				epsilon_min = epsilon_i;
				best_h = classifiers[i];
				i_minimisator = i;
			}
		}

		alfa = log((1.0-epsilon_min)/epsilon_min)/2;
		final_classifier[i_minimisator].w1 += alfa*best_h.w1;
		final_classifier[i_minimisator].w2 += alfa*best_h.w2;

		update_weight(weights_lambda, best_h, classifiers, features_of_images, i_minimisator, c, alfa);	
	}

	weights_lambda.clear();
	int size = filters.size();

	for(int i = 0; i < size; i++){
		features_of_images[i].clear();
	}

	features_of_images.clear();*/
}

int main () {

    auto start_haar_filter = std::chrono::high_resolution_clock::now();

    // create haar filters (1.2)
    std::vector<Haar_filter> filters;
    create_filters(filters);

    auto finish_haar_filter = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_haar_filter = finish_haar_filter - start_haar_filter;

    std::cout << "Elapsed Time (Haar Filter): " << elapsed_haar_filter.count() << std::endl;

    auto start_training = std::chrono::high_resolution_clock::now();

    // training Model (Ex 2.1)
    std::vector<Classifier> classifiers = initialize_classifier_vector(filters.size());
    train_model(classifiers, filters);

    auto finish_training = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_training = finish_training - start_training;

    std::cout << "Elapsed Time (Training): " << elapsed_training.count() << std::endl;
    std::cout << "filters size: " << filters.size() << std::endl;
    std::cout << "classifers size: " << classifiers.size() << std::endl;

    // Boosting des classifieurs faibles (Ex 2.2)
    std::vector<Classifier> final_classifier;
    boosting_classifiers(classifiers, filters, final_classifier);

    //TODO: lembrar de deletar todos os vetores

    filters.clear();
    classifiers.clear();
    final_classifier.clear();
    return 0;

}