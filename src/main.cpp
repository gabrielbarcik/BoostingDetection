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
#include <time.h>
#include <stdlib.h>

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
    int NUMBER_FILES_POS = number_of_files("/usr/local/INF442-2018/P5/app/pos/");
    int NUMBER_FILES_NEG = number_of_files("/usr/local/INF442-2018/P5/app/neg/");

    for (int i = 0; i < K; i++) {

        double r = ((double) rand() / (RAND_MAX));
        int r_int = 1;
        // before: r < 0.5
        // double threshold = (NUMBER_FILES_NEG) / (NUMBER_FILES_NEG + NUMBER_FILES_POS);
        double threshold = 0.5;
        if(r < threshold){
            cks[i] = -1;

            srand((unsigned)time(NULL));
            r = (rand()%(NUMBER_FILES_NEG));
            r_int = (int) r;

            random_img[i] = r_int;

        } else {
            cks[i] = 1;
    
            srand((unsigned)time(NULL));
            r = (rand()%(NUMBER_FILES_POS));
            r_int = (int) r;

            random_img[i] = r_int;
        }
    }
}

void create_features(int k, int* random_img, int* cks, int* features, double* classifiers_w1, double* classifiers_w2, int rank, 
                        int classifiers_size, std::vector<Haar_filter> &filters) {

    // std::cout << "begin of create_feature" << std::endl;

    std::string image_path;
    // neg folder
    if (cks[k] == -1) {
        image_path = "/usr/local/INF442-2018/P5/app/neg/im" + std::to_string(random_img[k]) + ".jpg";
    } else { // pos folder
        image_path = "/usr/local/INF442-2018/P5/app/pos/im" + std::to_string(random_img[k]) + ".jpg";
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

void train_model_mpi(int K, int* random_img, int* cks, double* classifiers_w1, double* classifiers_w2, int rank, int world_size, 
                        int classifiers_size, std::vector<Haar_filter> &filters){

    double epsilon = 0.001; // TODO: evaluate the right value for epsilon
    int* features = new int[classifiers_size];
    long Xki, h;

    double update_w1 = 0;
    double update_w2 = 0;
    double threshold_w1 = 0.0001;
    double threshold_w2 = 0.1;
    double max_update_w1 = -1;
    double max_update_w2 = -1;

    double global_max_update_w1;

    // std::cout << "Begin of train model mpi" << std::endl;

    for(int k = 0; k < K; k++){
        // auto start_random = std::chrono::high_resolution_clock::now();
        // std::cout << "getting feature" << std::endl;
        create_features(k, random_img, cks, features, classifiers_w1, classifiers_w2, rank, classifiers_size, filters);

        // auto finish_random = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed_random = finish_random - start_random;

        // auto start_classify = std::chrono::high_resolution_clock::now();
        max_update_w1 = -1;
        max_update_w2 = -1;
        global_max_update_w1 = -1;

        for(int i = 0; i < classifiers_size; i++){
            Xki = features[i];
            // std::cout << "getting h" << std::endl;
            h = classifier_h_of_a_feature(classifiers_w1[i], classifiers_w2[i], features[i]);
            update_w1 = epsilon * (h - cks[k]) * Xki;
            update_w2 = epsilon * (h - cks[k]);
            classifiers_w1[i] -= update_w1;
            classifiers_w2[i] -= update_w2;

            if (abs(update_w1) > max_update_w1) {
                max_update_w1 = abs(update_w1);
            }
            if (abs(update_w2) > max_update_w2) {
                max_update_w2 = abs(update_w2);
            }
        }

        // std::cout << "cartas" << std::endl;
        // std::cout << "in K = " << k << " max_update_w1 = " << max_update_w1 << std::endl;
        // std::cout << "in K = " << k << " max_update_w1 = " << max_update_w1 << std::endl;

        // std::cout << "calling reduce max update" << std::endl;
        MPI_Reduce(&max_update_w1, &global_max_update_w1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << "global max: " << global_max_update_w1 << std::endl;
            if (global_max_update_w1 < threshold_w1 && global_max_update_w1 != 0) {
                std::cout << "leaving training in k = " << k << std::endl;
                std::cout << "max_update_w1 = " << global_max_update_w1 << std::endl;
                break;
            }
        }

        // auto finish_classify = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed_classify = finish_classify - start_classify;
    }

}

void initialize_weights(double* weights_lambda, int n){
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
                    int i_minimizer, std::vector<int> &c, double alfa) {
	        
	int n = images.size();
    double sum = 0.0;
    calculate_feature_i(images, feature_i, filters, i_minimizer);
    // update weights
    for(int j = 0; j < n; j++){

        weights_lambda[j] *= exp(-c[j]*alfa*classifier_h_of_a_feature(global_classifier_w1[i_minimizer],
        	global_classifier_w2[i_minimizer], feature_i[j]));
        sum += weights_lambda[j];
    }

    // normalization
    for(int i = 0; i < n; i++){
        weights_lambda[i] /= sum;
    }
}


void boosting_classifiers(double* global_classifier_w1, double* global_classifier_w2,
						std::vector<Haar_filter> &filters, double* final_classifier_w1,
						double* final_classifier_w2, int rank, int world_size, int classifiers_size, 
                        std::vector<double> &alfa, std::vector<int> &strong_features_indices, int NUM_STRONG_FEATURES){
    
    // double teta = 0.789; // TODO: evaluete a good number for teta

	std::vector<int> c; // c[i] = 1 if the image is a face and -1 if it is not.
    std::vector<Image> images; // a vector of all images in dev/pos and dev/neg
    std::vector<int> feature_i;

    //initialization of the parameters
    initialize_images_dev(images);
    int NUM_IMAGES = images.size(); // the total number of images
    double* weights_lambda = new double[NUM_IMAGES];
    initialize_weights(weights_lambda, NUM_IMAGES);
    initialize_c_dev(c, NUM_IMAGES);
    initialize_feature_i_dev(feature_i, NUM_IMAGES);

    int features_size = filters.size(); // features_size is the total number of features of a image 92 x 112

	// find best h_i
	double epsilon_i; // one of the epsilons evalueted
    double epsilon_min = -1; // the smalest epsilon evalueted
    double my_epsilon_min = DBL_MAX;

	int *all_i_minimizer = new int [world_size];
    double* all_epsilons = new double[world_size];

    std::cout << "begin iterations: boosting function" << std::endl;

	for(int k = 0; k < NUM_STRONG_FEATURES; k++) {

    	int my_i_minimizer;
        int i_minimizer;
        my_epsilon_min = DBL_MAX;
        
        std::cout << "working between rank*classifiers_size and (rank+1)*classifier_size with " << rank << std::endl;

        for(int i = (rank)*classifiers_size; i < (rank+1)*classifiers_size; i++){
            epsilon_i = 0.0;
            calculate_feature_i(images, feature_i, filters, i);
            for(int j = 0; j < NUM_IMAGES; j++){
                epsilon_i += weights_lambda[j]*function_of_error_E(
                    classifier_h_of_a_feature(global_classifier_w1[i], global_classifier_w2[i], feature_i[j]),
                    c[j]);
            }

            if(epsilon_i < my_epsilon_min){
                my_epsilon_min = epsilon_i;
                my_i_minimizer = i;
            }
        }

        MPI_Gather(&my_i_minimizer, 1, MPI_INT, all_i_minimizer, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&my_epsilon_min, 1, MPI_DOUBLE, all_epsilons, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        std::cout << "gather done!" << std::endl;

        if (rank == 0) {
            epsilon_min = all_epsilons[0];
            i_minimizer = all_i_minimizer[0];
            for (int i = 0; i < world_size; i++) {
                if (epsilon_min > all_epsilons[i]){
                    epsilon_min = all_epsilons[i];
                    i_minimizer = all_i_minimizer[i];
                }
            }
            std::cout << "error min = " << epsilon_min << std::endl;

            if (epsilon_min < 0.1) {
                std::cout << "algorithm converged, error min = " << epsilon_min << std::endl;
                break; 
            }

            alfa.push_back(log((1.0-epsilon_min)/epsilon_min)/2);
            strong_features_indices.push_back(i_minimizer);
			final_classifier_w1[i_minimizer] += alfa.back()*global_classifier_w1[i_minimizer];
			final_classifier_w2[i_minimizer] += alfa.back()*global_classifier_w2[i_minimizer];
			update_weight(weights_lambda, global_classifier_w1, global_classifier_w2 , images, feature_i, filters, i_minimizer, c, alfa.back());	
        }

 		MPI_Bcast(weights_lambda, NUM_IMAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
    
    
	delete[] weights_lambda;
	images.clear();
	feature_i.clear();
	c.clear();
    delete[] all_i_minimizer;
    delete[] all_epsilons;
}

int main (int argc, char* argv[]) {

    // srand (time(NULL));

    const int root = 0;
    int rank, world_size;
    
    // Launch MPI processes on each node
    MPI_Init(&argc, &argv);

/*
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        // Error - MPI does not provide needed threading level
        std::cout << "ERROR" << std::endl;
    }
*/

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
    
    // PARAMETERS
    const int K = 10000; // TODO: evaluate the right value for K

    int* random_img = new int[K]; // index of random image
    int* cks = new int[K]; // gives the origin of this image
    double* global_classifier_w1 = new double[NUM_CLASSIFIERS];
    double* global_classifier_w2 = new double[NUM_CLASSIFIERS];

    if (rank == root) {
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
    } else {
        classifiers_size = div;
        classifiers_w1 = new double[classifiers_size];
        classifiers_w2 = new double[classifiers_size];
    }

    for (int i = 0; i < classifiers_size; i++) {
        classifiers_w1[i] = 1;
        classifiers_w2[i] = 0;
    }

    std::cout << "before train mpi" << std::endl;

    train_model_mpi(K, random_img, cks, classifiers_w1, classifiers_w2, rank, world_size, classifiers_size, filters);

    std::cout << "end of train model... before gather" << std::endl;

    MPI_Gather(classifiers_w1, classifiers_size, MPI_DOUBLE, global_classifier_w1, classifiers_size, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Gather(classifiers_w2, classifiers_size, MPI_DOUBLE, global_classifier_w2, classifiers_size, MPI_DOUBLE, root, MPI_COMM_WORLD);

    std::cout << "end of weak training" << std::endl; 

    auto finish_training = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_training = finish_training - start_training;
    
    if (rank == root) {
        std::cout << "Elapsed Time (Training): " << elapsed_training.count() << std::endl;
        ofstream myfile;
		myfile.open ("global_classifier_w1_small_200.txt");
		for (int i = 0; i < NUM_CLASSIFIERS; i++){
			myfile << global_classifier_w1[i] << "\n";
		}
		myfile.close();

        ofstream myfile2;
        myfile2.open ("global_classifier_w2_small_200.txt");
		for (int i = 0; i < NUM_CLASSIFIERS; i++){
			myfile2 << global_classifier_w2[i] << "\n";
		}
		myfile2.close();
    }

    // Ex 2.2

    auto start_boosting = std::chrono::high_resolution_clock::now();

    // Boosting des classifieurs faibles (Ex 2.2)
    double* final_classifier_w1;
    double* final_classifier_w2;
    int NUM_STRONG_FEATURES = 200; // TODO: find out
    std::vector<double> alfa;
    std::vector<int> strong_features_indices;
    
    final_classifier_w1 = new double[NUM_CLASSIFIERS]; // the defeult value is already 0 for all positions
    final_classifier_w2 = new double[NUM_CLASSIFIERS];
    
       // Preciso dar um Bcast de global_classifier_w1 e w2 pra todo mundo
    MPI_Bcast(global_classifier_w1, NUM_CLASSIFIERS, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Bcast(global_classifier_w2, NUM_CLASSIFIERS, MPI_DOUBLE, root, MPI_COMM_WORLD);
    std::cout << "broadcast of global_classifier done correctly!" << std::endl;
 
    boosting_classifiers(global_classifier_w1, global_classifier_w2 ,filters, 
    	final_classifier_w1, final_classifier_w2, rank, world_size, classifiers_size, alfa, strong_features_indices, NUM_STRONG_FEATURES);

    auto finish_boosting = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_boosting = finish_boosting - start_boosting;

    std::cout << "Elapsed Time (Boosting): " << elapsed_boosting.count() << std::endl;    

    filters.clear();
    delete[] random_img;
    delete[] cks;
    delete[] global_classifier_w1;
    delete[] global_classifier_w2;

    // Save the final classifier in a file
    if(rank == root){

    	ofstream myfile;
		myfile.open ("final_classifier_w1_small_200.txt");
		for (int i = 0; i < NUM_CLASSIFIERS; i++){
			myfile << final_classifier_w1[i] << "\n";
		}
		myfile.close();

		myfile.open ("final_classifier_w2_small_200.txt");
		for (int i = 0; i < NUM_CLASSIFIERS; i++){
			myfile << final_classifier_w2[i] << "\n";
		}
		myfile.close();

        myfile.open("alpha_small_200.txt");
        for (int i = 0; i < NUM_STRONG_FEATURES; i++){
			myfile << alfa[i] << "\n";
		}
		myfile.close();

        myfile.open("indices_final_classifier_small_200.txt");
        for (int i = 0; i < NUM_STRONG_FEATURES; i++){
			myfile << strong_features_indices[i] << "\n";
		}
		myfile.close();

    }

    delete[] final_classifier_w1;
    delete[] final_classifier_w2;
    alfa.clear();
    strong_features_indices.clear();

    MPI_Finalize();

    return 0;

}