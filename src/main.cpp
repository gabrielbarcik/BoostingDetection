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

#include <fstream>
using namespace std;


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

int classifier_h_of_a_feature (double w1, double w2, int feature_of_image) {
	if (w1 * (double) feature_of_image + w2 >= 0.0)
		return 1;
	else 
		return -1;
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
            h = classifier_h_of_a_feature(classifiers_w1[i], classifiers_w2[i], features[i]);
            classifiers_w1[i] -= epsilon * (h - cks[k]) * Xki;
            classifiers_w2[i] -= epsilon * (h - cks[k]);
        }
        // auto finish_classify = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed_classify = finish_classify - start_classify;
    }
}

void initialize_weights(double* weights_lambda, int n){
	weights_lambda = new double[n];
	double base_case = 1 / (double) n;
	for(unsigned long i = 0; i < n; i++){
		weights_lambda[i] = base_case;
	}
}

int function_of_error_E (int h, int c){
	if (h == c){
		return 0;
	} else
		return 1;
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

void update_weight(double* weights_lambda, double* global_classifier_w1, double* global_classifier_w2, 
					std::vector<Image> &images, std::vector<int> &feature_i, std::vector<Haar_filter> &filters, 
                    int i_minimisator, std::vector<int> &c, double alfa) {
	        
	int n = images.size();
    double sum = 0.0;
    calculate_feature_i(images, feature_i, filters, i_minimisator);
    // update weights
    for(int j = 0; j < n; j++){

        weights_lambda[j] *= exp(-c[j]*alfa*classifier_h_of_a_feature(global_classifier_w1[i_minimisator],
        	global_classifier_w2[i_minimisator], feature_i[j]));
        sum += weights_lambda[j];
    }

    // normalization
    for(int i = 0; i < n; i++){
        weights_lambda[i] /= sum;
    }
}


void boosting_classifiers(double* global_classifier_w1, double* global_classifier_w2,
						std::vector<Haar_filter> &filters, double* final_classifier_w1,
						double* final_classifier_w2, int rank, int world_size, int classifiers_size){

    double* weights_lambda;
	std::vector<int> c; // c[i] = 1 if the image is a face and -1 if it is not.
    std::vector<Image> images; // a vector of all images in dev/pos and dev/neg
    std::vector<int> feature_i;

    //initialization of the parameters
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
	int *all_i_minimisateurs = new int [world_size];
    int num_interaction_N = 10; // TODO: evaluete a good number for N    


	for(int k = 0; k < num_interaction_N; k++) {

        int div = features_size / world_size;
    	int rest = features_size % world_size;
    	int my_i_minimisator;
        double my_epsilon_min = DBL_MAX;
        if (rank != world_size - 1){
        	for(int i = rank*div; i < (rank+1)*div; i++){
	            epsilon_i = 0.0;
	            calculate_feature_i(images, feature_i, filters, i);
	            for(int j = 0; j < num_images; j++){
	                epsilon_i += weights_lambda[j]*function_of_error_E(
	                    classifier_h_of_a_feature(global_classifier_w1[i], global_classifier_w2[i], feature_i[j]),
	                    c[j]);
            	}

	            if(epsilon_i < my_epsilon_min){
	                my_epsilon_min = epsilon_i;
	                my_i_minimisator = i;
	            }

        	}
        } else {
        	std::cout << "Hi1" << std::endl;
        	for(int i = rank*div; i < (rank+1)*div + rest; i++){
	            epsilon_i = 0.0;
	            std::cout << "Hi2" << std::endl;
	            calculate_feature_i(images, feature_i, filters, i);
	            std::cout << "Hi3" << std::endl;
	            for(int j = 0; j < num_images; j++){
	                epsilon_i += weights_lambda[j]*function_of_error_E(
	                    classifier_h_of_a_feature(global_classifier_w1[i], global_classifier_w2[i], feature_i[j]),
	                    c[j]);
            	}
            	std::cout << "Hi4" << std::endl;
	            if(epsilon_i < my_epsilon_min){
	                my_epsilon_min = epsilon_i;
	                my_i_minimisator = i;
	            }

        	}
        }

		
        MPI_Reduce(&my_epsilon_min, &epsilon_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Gather(&my_i_minimisator, 1, MPI_INT, all_i_minimisateurs, world_size, MPI_INT, 0, MPI_COMM_WORLD);

 		
 		if(rank == 0){
 			int i_minimisator; // the positions of the features that minimizes the erreur
 			
 			for(int k = 0; k < world_size; k++){ // find i_minimisator
 				int i = all_i_minimisateurs[k];
 				epsilon_i = 0.0;
	            calculate_feature_i(images, feature_i, filters, i);
	            for(int j = 0; j < num_images; j++){
	                epsilon_i += weights_lambda[j]*function_of_error_E(
	                    classifier_h_of_a_feature(global_classifier_w1[i], global_classifier_w2[i], feature_i[j]),
	                    c[j]);
 				}
 				if(epsilon_i == epsilon_min){
 					i_minimisator = i;
 					break;
 				}
 			}


 			std::cout << "epsilon_min = " << epsilon_min << " | i_minimisator = " << i_minimisator << std::endl;

			alfa = log((1.0-epsilon_min)/epsilon_min)/2;
		
			final_classifier_w1[i_minimisator] += alfa*global_classifier_w1[i_minimisator];
			final_classifier_w2[i_minimisator] += alfa*global_classifier_w2[i_minimisator];
			update_weight(weights_lambda, global_classifier_w1, global_classifier_w2 , images, feature_i, filters, i_minimisator, c, alfa);	
 		}

 		MPI_Bcast(weights_lambda, num_images, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		
	}

	delete[] weights_lambda;
	images.clear();
	feature_i.clear();
	c.clear();

}

int main (int argc, char* argv[]) {
    // PARAMETERS
    int K = 4; // TODO: evaluate the right value for K
    
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


    // ********************************************************************************************************************************************************
    // Ex 2.2
    // TODO PARALELIZAR ABAIXO

    auto start_boosting = std::chrono::high_resolution_clock::now();

    // Boosting des classifieurs faibles (Ex 2.2)
    double* final_classifier_w1;
    double* final_classifier_w2;

    //if (rank == root) {
        final_classifier_w1 = new double[NUM_CLASSIFIERS]; // the defeult value is already 0 for all positions
        final_classifier_w2 = new double[NUM_CLASSIFIERS];
    //}

       // Preciso dar um Bcast de global_classifier_w1 e w2 pra todo mundo
   std::cout << "Ola" << std::endl;
   MPI_Bcast(global_classifier_w1, NUM_CLASSIFIERS, MPI_DOUBLE, root, MPI_COMM_WORLD);
   MPI_Bcast(global_classifier_w2, NUM_CLASSIFIERS, MPI_DOUBLE, root, MPI_COMM_WORLD);
   std::cout << "Oi" << std::endl;
   /*if(rank == 1){
   		for(int i = 0; i <= 0; i++)
   		std::cout << "w1 = " << global_classifier_w1[i]  << " w2 = " << global_classifier_w2[i] << std::endl;

   }*/

   /*boosting_classifiers(global_classifier_w1, global_classifier_w2 ,filters, 
    	final_classifier_w1, final_classifier_w2, rank, world_size, classifiers_size);
*/
    auto finish_boosting = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_boosting = finish_boosting - start_boosting;

    std::cout << "Elapsed Time (Boosting): " << elapsed_boosting.count() << std::endl;    


    filters.clear();
    delete[] random_img;
    delete[] cks;
    
    if(rank == root){
    	delete[] global_classifier_w1;
    	delete[] global_classifier_w2;
    }


    // Save the final classifier in a file
    if(rank == root){

    	ofstream myfile;
		myfile.open ("final_classifier_w1.txt");
		for (int i = 0; i < NUM_CLASSIFIERS; i++){
			myfile << final_classifier_w1[i] << "\n";
		}
		myfile.close();

		myfile.open ("final_classifier_w2.txt");
		for (int i = 0; i < NUM_CLASSIFIERS; i++){
			myfile << final_classifier_w2[i] << "\n";
		}
		myfile.close();
    }


    MPI_Finalize();


    return 0;

}