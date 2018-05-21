#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.hpp>
#include <opencv2/imgproc.hpp>
#include "../lib/image.hpp"
#include "../lib/haar_filter.hpp"
#include "../lib/classifier.hpp"
#include <chrono>
#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>

#include <string>

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
int number_of_files(std::string folder_name){
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
        // std::cout << "Elapsed Time (Random Function): " << elapsed_random.count() << std::endl;

        auto start_classify = std::chrono::high_resolution_clock::now();
        for(unsigned long i = 0; i < features.size(); i++){
            Xki = features[i];

            if (classifiers[i].w1 * Xki + classifiers[i].w2 >= 0.0)
                h = 1;
            else 
            	h = -1;
            classifiers[i].w1 -= epsilon * (h - ck) * Xki;
            classifiers[i].w2 -= epsilon * (h - ck);
        }
        auto finish_classify = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_classify = finish_classify - start_classify;
        // std::cout << "Elapsed Time (Classify): " << elapsed_classify.count() << std::endl;
        features.clear();
    }
}

std::vector<Classifier> initialize_classifier_vector(unsigned long size){
	std::vector<Classifier> vec;

	for(unsigned long i = 0; i < size; i++)
		vec.push_back(Classifier());

	return vec;
}


int main () {

    auto start_haar_filter = std::chrono::high_resolution_clock::now();

    // create haar filters
    std::vector<Haar_filter> filters;
    create_filters(filters);

    auto finish_haar_filter = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_haar_filter = finish_haar_filter - start_haar_filter;

    std::cout << "Elapsed Time (Haar Filter): " << elapsed_haar_filter.count() << std::endl;

    auto start_training = std::chrono::high_resolution_clock::now();

    // training Model
    std::vector<Classifier> classifiers = initialize_classifier_vector(filters.size());
    train_model(classifiers, filters);

    auto finish_training = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_training = finish_training - start_training;

    std::cout << "Elapsed Time (Training): " << elapsed_training.count() << std::endl;
    std::cout << "filters size: " << filters.size() << std::endl;
    std::cout << "classifers size: " << classifiers.size() << std::endl;

    return 0;

}