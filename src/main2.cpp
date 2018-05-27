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
// #include "/usr/local/boost-1.64/"


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

        srand((unsigned)time(NULL));
        r = (rand()%(NUMBER_FILES_NEG));
        r_int = (int) r;

        image_path = "/usr/local/INF442-2018/P5/app/neg/im" + std::to_string(r_int) + ".jpg";

        
    } else {
        ck = 1;

        srand((unsigned)time(NULL));
        r = (rand()%(NUMBER_FILES_POS));
        r_int = (int) r;

        image_path = "/usr/local/INF442-2018/P5/app/pos/im" + std::to_string(r_int) + ".jpg";
    }

    cv::Mat image = cv::imread(image_path);
    if (r_int == 1)
        cv::imshow("image", image);
    Image img = Image(image_path, image);

    auto start_feature = std::chrono::high_resolution_clock::now();
    /*for (unsigned long i = 0; i < filters.size(); i++) {
        features.push_back(filters[i].feature(img.integral_image));
    }*/
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
    int NUMBER_FILES_POS = number_of_files("/usr/local/INF442-2018/P5/app/pos/");
    // int NUMBER_FILES_POS = number_of_files("pos/"); // meu pc
    int NUMBER_FILES_NEG = number_of_files("/usr/local/INF442-2018/P5/app/neg/");
    // int NUMBER_FILES_NEG = number_of_files("neg/"); // meu pc

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

void initialize_weights(std::vector<double> &weights_lambda, int n){
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

void initialize_F(std::vector<Classifier> &classifiers_result_F, int size){
	for (int i = 0; i < size; i++){
		classifiers_result_F.push_back(Classifier(0.0,0.0));
	}
}

void initialize_images_dev (std::vector<Image> &images){
    int NUMBER_FILES_POS = number_of_files("/usr/local/INF442-2018/P5/dev/pos");
    int NUMBER_FILES_NEG = number_of_files("/usr/local/INF442-2018/P5/dev/neg");

    std::string image_path;
    cv::Mat image;

    for(int i = 0; i < NUMBER_FILES_NEG; i++){
        image_path = "/usr/local/INF442-2018/P5/dev/neg/im" + std::to_string(i) + ".jpg";
        image = cv::imread(image_path);
        images.push_back(Image(image_path, image));
    }
       
    for(int i = 0; i < NUMBER_FILES_POS; i++){
        image_path = "/usr/local/INF442-2018/P5/dev/pos/im" + std::to_string(i) + ".jpg";
        image = cv::imread(image_path);
        images.push_back(Image(image_path, image));
    }
}

void initialize_c_dev(std::vector<int> &c, int n){
    int NUMBER_FILES_NEG = number_of_files("/usr/local/INF442-2018/P5/dev/neg");
    
    for(int i = 0; i < n; i++){
        if (i < NUMBER_FILES_NEG)
            c.push_back(-1);
        else
            c.push_back(1);
    }
}

void initialize_feature_i_dev(std::vector<int> &feature_i, int n){
    for(int i = 0; i < n; i++){
        feature_i.push_back(0);
    }
}

void calculate_feature_i(std::vector<Image> &images, std::vector<int> &feature_i, 
                         std::vector<Haar_filter> &filters, int feature_number){
    for(int pos_img = 0; pos_img < images.size(); pos_img++){
        feature_i[pos_img] = filters[feature_number].feature(images[pos_img].integral_image);
    }

}

void update_weight(std::vector<double> &weights_lambda, Classifier best_h, 
                    std::vector<Classifier> &classifiers, std::vector<Image> &images, 
                    std::vector<int> &feature_i, std::vector<Haar_filter> &filters, 
                    int i_minimisator, std::vector<int> &c, double alfa) {
    double sum = 0.0;
    calculate_feature_i(images, feature_i, filters, i_minimisator);
    // update weights
    for(int j = 0; j < images.size(); j++){

        weights_lambda[j] *= exp(-c[j]*alfa*classifier_h_of_a_feature(best_h, feature_i[j]));
        sum += weights_lambda[j];
    }

    // normalization
    for(int i = 0; i < weights_lambda.size(); i++){
        weights_lambda[i] /= sum;
    }
}

void boosting_classifiers(std::vector<Classifier> &classifiers, std::vector<Haar_filter> &filters,
									std::vector<Classifier> &final_classifier){


    
    // double teta = 0.789; // TODO: evaluete a good number for teta
    

    //TODO: calcula pra uma features i, um vetor e vou escrever em cima dele
    // std::vector<int> feature_i -> feature_i[k] = filters[i].calculate_feature(images[k].integral_image)

	// std::vector< std::vector<int> > features_of_images;
    std::vector<double> weights_lambda;
	std::vector<int> c; // c[i] = 1 if the image is a face and -1 if it is not.
    std::vector<Image> images; // a vector of all images in dev/pos and dev/neg
    std::vector<int> feature_i;

    initialize_F(final_classifier, classifiers.size());
    initialize_images_dev(images);
    initialize_weights(weights_lambda, images.size());
    initialize_c_dev(c, images.size());
    initialize_feature_i_dev(feature_i, images.size());

    int num_images = images.size(); // n is the total number of images
    int features_size = filters.size(); // features_size is the total number of features of a image 92 x 112

	// find best h_i
	double epsilon_i; // one of the epsilons evalueted
    double epsilon_min; // the smalest epsilon evalueted
    double alfa; // alfa(k) = ln ((1 - episilon(k))/k) / 2
	Classifier best_h; // is the classifier that minimises the erreur (epsilon)
	int i_minimisator; // the positions of the features that minimizes the erreur
    int num_interaction_N = 1; // TODO: evaluete a good number for N


	for(int k = 0; k < num_interaction_N; k++) {
        epsilon_min = DBL_MAX;
        
        for(int i = 0; i < features_size; i++){
            epsilon_i = 0.0;
            std::cout << "here k = " << k << "/" << num_interaction_N << " | i = " << i << std::endl;
            calculate_feature_i(images, feature_i, filters, i);
            for(int j = 0; j < num_images; j++){
                epsilon_i += weights_lambda[j]*function_of_error_E(
                    classifier_h_of_a_feature(classifiers[i], feature_i[j]),
                    c[j]);
            }

            if(epsilon_i < epsilon_min){
                epsilon_min = epsilon_i;
                best_h = classifiers[i];
                i_minimisator = i;
            }

        }


		std::cout << "epsilon_min = " << epsilon_min << " | i_minimisator = " << i_minimisator << std::endl;

		alfa = log((1.0-epsilon_min)/epsilon_min)/2;
		final_classifier[i_minimisator].w1 += alfa*best_h.w1;
		final_classifier[i_minimisator].w2 += alfa*best_h.w2;
        std::cout << "debug1" << std::endl;
		update_weight(weights_lambda, best_h, classifiers, images, feature_i, filters, i_minimisator, c, alfa);	
        std::cout << "debug2" << std::endl;
	}

	weights_lambda.clear();
	int size = filters.size();
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






	auto start_boosting = std::chrono::high_resolution_clock::now();

    // Boosting des classifieurs faibles (Ex 2.2)
    std::vector<Classifier> final_classifier;
    boosting_classifiers(classifiers, filters, final_classifier);

    auto finish_boosting = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_boosting = finish_boosting - start_boosting;

    std::cout << "Elapsed Time (Boosting): " << elapsed_boosting.count() << std::endl;    





    std::cout << "filters size: " << filters.size() << std::endl;
    std::cout << "classifers size: " << classifiers.size() << std::endl;

    //TODO: lembrar de deletar todos os vetores
    for(int i = 0; i < 40 ; i++){
    	if(final_classifier[i].w1 != 0.0)
    		final_classifier[i].print();
    }


    filters.clear();
    classifiers.clear();
    final_classifier.clear();
    return 0;

}