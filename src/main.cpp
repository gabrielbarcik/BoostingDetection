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
        // image_path = "/users/eleves-a/2017/gabriel.fedrigo-barcik/BoostingDetection/build/neg/";
        srand((unsigned)time(NULL));
        r = (rand()%(NUMBER_FILES_NEG)) + 1.0;
        r_int = (int) r;

        image_path = "neg/im" + std::to_string(r_int) + ".jpg";
        // image_path = "/users/eleves-a/2017/gabriel.fedrigo-barcik/BoostingDetection/build/neg/im" + std::to_string(r_int) + ".jpg";
        // std::cout << image_path << std::endl;
        
    } else {
        ck = 1;
        image_path = "pos/";
        // image_path = "/users/eleves-a/2017/gabriel.fedrigo-barcik/BoostingDetection/build/pos/";
        srand((unsigned)time(NULL));
        r = (rand()%(NUMBER_FILES_POS)) + 1.0;
        r_int = (int) r;

        image_path = "pos/im" + std::to_string(r_int) + ".jpg";
        // image_path = "/users/eleves-a/2017/gabriel.fedrigo-barcik/BoostingDetection/build/pos/im" + std::to_string(r_int) + ".jpg";
  
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


void generate_random_vectors(int* random_img, int* cks, int K) {
    // int NUMBER_FILES_POS = number_of_files("/users/eleves-a/2017/gabriel.fedrigo-barcik/BoostingDetection/build/pos/");
    int NUMBER_FILES_POS = number_of_files("/usr/local/INF442-2018/P5/app/pos/");
    // int NUMBER_FILES_NEG = number_of_files("/users/eleves-a/2017/gabriel.fedrigo-barcik/BoostingDetection/build/neg/");
    int NUMBER_FILES_NEG = number_of_files("/usr/local/INF442-2018/P5/app/neg/");
    for (int i = 0; i < K; i++) {

        double r = ((double) rand() / (RAND_MAX));
        int r_int = 1;

        if(r < 0.5){
            cks[i] = -1;

            // srand((unsigned)time(NULL));
            r = (rand()%(NUMBER_FILES_NEG));
            r_int = (int) r;

            random_img[i] = r_int;

        } else {
            cks[i] = 1;
    
            // srand((unsigned)time(NULL));
            r = (rand()%(NUMBER_FILES_POS));
            r_int = (int) r;

            random_img[i] = r_int;
        }
    }
}

void create_features(int k, int* random_img, int* cks, int* features, double* classifiers_w1, double* classifiers_w2, int rank, int classifiers_size, std::vector<Haar_filter> &filters) {

    // std::cout << "begin of create_feature" << std::endl;

    std::string image_path;
    // neg folder
    if (cks[k] == -1) {
        image_path = "/usr/local/INF442-2018/P5/app/neg/im" + std::to_string(random_img[k]) + ".jpg";
        // image_path = "/users/eleves-a/2017/gabriel.fedrigo-barcik/BoostingDetection/build/neg/im" + std::to_string(random_img[k]) + ".jpg";.
    } else { // pos folder
        image_path = "/usr/local/INF442-2018/P5/app/pos/im" + std::to_string(random_img[k]) + ".jpg";
        // image_path = "/users/eleves-a/2017/gabriel.fedrigo-barcik/BoostingDetection/build/neg/im" + std::to_string(random_img[k]) + ".jpg";.
    }

    // std::cout << "ck: " << cks[k] << " random nb: " << random_img[k] << std::endl;

    cv::Mat image = cv::imread(image_path);
    Image img = Image(image_path, image);

    // std::cout << "image found" << std::endl;

    int index = 0;
    for (unsigned long i = rank*classifiers_size; i < (rank+1)*classifiers_size; i++) {
        features[index] = filters[i].feature(img.integral_image);
        index++;
    }

}

int classifier_h_of_a_feature (Classifier classifier, double feature_of_image) {
	if (classifier.w1 * feature_of_image + classifier.w2 >= 0.0)
		return 1;
	else 
		return -1;
}

int classifier_h_of_a_feature_mpi (double w1, double w2, int feature) {
	if (w1 * feature + w2 >= 0.0)
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
    // int NUMBER_FILES_POS = number_of_files("/users/eleves-a/2017/gabriel.fedrigo-barcik/BoostingDetection/build/pos/");
    int NUMBER_FILES_NEG = number_of_files("neg/");
    // int NUMBER_FILES_POS = number_of_files("/users/eleves-a/2017/gabriel.fedrigo-barcik/BoostingDetection/build/neg/");

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

void train_model_mpi(int K, int* random_img, int* cks, double* classifiers_w1, double* classifiers_w2, int rank, int world_size, int classifiers_size, std::vector<Haar_filter> &filters){
    double epsilon = 0.01; // TODO: evaluate the right value for epsilon
    int* features = new int[classifiers_size];
    long Xki, h;

    // std::cout << "Begin of train model mpi" << std::endl;

    for(int k = 0; k < K; k++){
        // auto start_random = std::chrono::high_resolution_clock::now();
        // std::cout << "getting feature" << std::endl;
        create_features(k, random_img, cks, features, classifiers_w1, classifiers_w2, rank, classifiers_size, filters);

        // auto finish_random = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed_random = finish_random - start_random;

        // auto start_classify = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < classifiers_size; i++){
            Xki = features[i];
            h = classifier_h_of_a_feature_mpi(classifiers_w1[i], classifiers_w2[i], features[i]);
            classifiers_w1[i] -= epsilon * (h - cks[k]) * Xki;
            classifiers_w2[i] -= epsilon * (h - cks[k]);
        }
        // auto finish_classify = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed_classify = finish_classify - start_classify;
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

int main (int argc, char* argv[]) {
    // PARAMETERS
    int K = 1000; // TODO: evaluate the right value for K
    
    const int root = 0;
    int rank, world_size;
    
    // Launch MPI processes on each node
    MPI_Init(&argc, &argv);

    // Get the id and number of threads
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::cout << "hello MPI user: from process " << rank << " of " << world_size << std::endl;

    // auto start_haar_filter = std::chrono::high_resolution_clock::now();

    // Create haar filters (1.2)
    std::vector<Haar_filter> filters;
    create_filters(filters);
    unsigned long NUM_CLASSIFIERS = filters.size();

    std::cout << "filters have been created" << std::endl;

    // auto finish_haar_filter = std::chrono::high_resolution_clock::now();

    // std::chrono::duration<double> elapsed_haar_filter = finish_haar_filter - start_haar_filter;

    // std::cout << "Elapsed Time (Haar Filter): " << elapsed_haar_filter.count() << std::endl;

    auto start_training = std::chrono::high_resolution_clock::now();

    // training Model (Ex 2.1)
    // std::vector<Classifier> classifiers = initialize_classifier_vector(filters.size());

    // Create variables for selecting the same images

    int* random_img = new int[K]; // index of random image
    int* cks = new int[K]; // gives the origin of this image
    double* global_classifier_w1;
    double* global_classifier_w2;

    if (rank == root) {
        global_classifier_w1 = new double[NUM_CLASSIFIERS];
        global_classifier_w2 = new double[NUM_CLASSIFIERS];
        // pick-up random images
        std::cout << "generate random vectors" << std::endl;
        generate_random_vectors(random_img, cks, K);
    }

    std::cout << "begin broadcast by " << rank << std::endl;

    // Broadcast the chosen images to all process
    MPI_Bcast(random_img, K, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(cks, K, MPI_INT, root, MPI_COMM_WORLD);


    // Create a part of the classifiers
    int div = NUM_CLASSIFIERS / world_size;
    int rest = NUM_CLASSIFIERS % world_size;
    double* classifiers_w1;
    double* classifiers_w2;
    int classifiers_size;
    if (rank == world_size - 1) {
        classifiers_size = div + rest;
        classifiers_w1 = new double[classifiers_size];
        classifiers_w2 = new double[classifiers_size];
        std::cout << "calling mpi train from last process" << std::endl;
        train_model_mpi(K, random_img, cks, classifiers_w1, classifiers_w2, rank, world_size, classifiers_size, filters);
    } else {
        classifiers_size = div;
        classifiers_w1 = new double[classifiers_size];
        classifiers_w2 = new double[classifiers_size];
        std::cout << "calling mpi train from all others processes" << std::endl;
        train_model_mpi(K, random_img, cks, classifiers_w1, classifiers_w2, rank, world_size, classifiers_size, filters);
    }

    std::cout << "end of train model... before gather" << std::endl;

    MPI_Gather(classifiers_w1, classifiers_size, MPI_DOUBLE, global_classifier_w1, classifiers_size, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Gather(classifiers_w2, classifiers_size, MPI_DOUBLE, global_classifier_w2, classifiers_size, MPI_DOUBLE, root, MPI_COMM_WORLD);

    std::cout << "end of weak training" << std::endl; 

    auto finish_training = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_training = finish_training - start_training;
    
    if (rank == root)
        std::cout << "Elapsed Time (Training): " << elapsed_training.count() << std::endl;

    // std::cout << "filters size: " << filters.size() << std::endl;
    // std::cout << "classifers size: " << classifiers.size() << std::endl;

    // Boosting des classifieurs faibles (Ex 2.2)
    // std::vector<Classifier> final_classifier;
    // boosting_classifiers(classifiers, filters, final_classifier);

    //TODO: lembrar de deletar todos os vetores

    // filters.clear();
    // classifiers.clear();
    // final_classifier.clear();

    MPI_Finalize();

    return 0;

}