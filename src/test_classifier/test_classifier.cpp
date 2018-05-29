#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../../lib/image.hpp"
#include "../../lib/haar_filter.hpp"

#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>

#include <string>
#include <float.h>

#include <mpi.h> 

#include <fstream>
using namespace std;
#include <vector>

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

void initialize_images_test (std::vector<Image> &images, string path){
    int NUMBER_FILES = number_of_files("/usr/local/INF442-2018/P5/test/" + path);

    std::string image_path;
    cv::Mat image;

    for(int i = 0; i < NUMBER_FILES; i++){
        image_path = "/usr/local/INF442-2018/P5/test/" + path + "/im" + std::to_string(i) + ".jpg";
        image = cv::imread(image_path);
        images.push_back(Image(image_path, image));
    }
}

void initialize_test_final_classifier_w1(std::vector<double> &final_classifier_w1){
    std::ifstream ifile("final_classifier_w1_small3.txt", std::ios::in);

    //check to see that the file was opened correctly:
    if (!ifile.is_open()) {
        std::cout << "deu ruim classifier w1" << std::endl;
        std::cerr << "There was a problem opening the input file final_classifier_w1_200.txt!\n";
        exit(1);//exit or do additional error checking
    }

    double num = 0.0;
    //keep storing values from the text file so long as data exists:
    while (ifile >> num) {
        final_classifier_w1.push_back(num);
    }
}

void initialize_test_final_classifier_w2(std::vector<double> &final_classifier_w2){
    std::ifstream ifile("final_classifier_w2_small3.txt", std::ios::in);

    //check to see that the file was opened correctly:
    if (!ifile.is_open()) {
        std::cout << "deu ruim classifier w2" << std::endl;
        std::cerr << "There was a problem opening the input file final_classifier_w2_200.txt!\n";
        exit(1);//exit or do additional error checking
    }

    double num = 0.0;
    //keep storing values from the text file so long as data exists:
    while (ifile >> num) {
        final_classifier_w2.push_back(num);
    }
}

void initialize_test_alfas(std::vector<double> &alfas){
    std::ifstream ifile("alpha_small3.txt", std::ios::in);

    //check to see that the file was opened correctly:
    if (!ifile.is_open()) {
        std::cout << "deu ruim alpha" << std::endl;
        std::cerr << "There was a problem opening the input file alpha_200.txt!\n";
        exit(1);//exit or do additional error checking
    }

    double num = 0.0;
    //keep storing values from the text file so long as data exists:
    while (ifile >> num) {
        alfas.push_back(num);
    }
}

void initialize_test_strong_features_indices(std::vector<int> &strong_features_indices){
    std::ifstream ifile("indices_final_classifier_small3.txt", std::ios::in);

    //check to see that the file was opened correctly:
    if (!ifile.is_open()) {
        std::cout << "deu ruim strong" << std::endl;
        std::cerr << "There was a problem opening the input file strong_features_indices.txt!\n";
        exit(1);//exit or do additional error checking
    }

    double num = 0.0;
    //keep storing values from the text file so long as data exists:
    while (ifile >> num) {
        strong_features_indices.push_back(num);
    }
}

int f_of_x(std::vector<double> &final_classifier_w1, std::vector<double> &final_classifier_w2, 
    std::vector<Haar_filter> &filters, std::vector<int> &strong_features_indices, Image img, double lim){

    double sum = 0;
    for(int i = 0; i < strong_features_indices.size(); i++){
        int index = strong_features_indices[i];
        sum += filters[index].feature(img.integral_image)*final_classifier_w1[index] + final_classifier_w2[index];
    }

    if(sum >= lim)
        return 1;
    else
        return -1;
}

void test_positives(int* correct_positive, int* false_negative, int position, std::vector<Image> &positive_images, 
    std::vector<Haar_filter> &filters, std::vector<double> &final_classifier_w1, std::vector<double> &final_classifier_w2, 
    std::vector<double> &alfas, std::vector<int> &strong_features_indices, double theta, int rank){

    double lim = 0.0;

    for(int i = 0; i < alfas.size(); i++)
        lim += alfas[i];

    lim *= theta;

    std::cout << "test_positives" << std::endl;

    for(int i = 0; i < positive_images.size(); i++){
        double f_x = f_of_x(final_classifier_w1, final_classifier_w2, filters, strong_features_indices, positive_images[i], lim);
        if (f_x == 1){
            correct_positive[position]++;
        } else if (f_x == -1){
            false_negative[position]++;
        }
        if (rank == 0) {
            std::cout << "positives_images = " << i << "/" <<  positive_images.size() << 
            "  |  theta = " << theta << std::endl;
        }

    }

}

void test_negatives(int* correct_negative, int* false_positive, int position, std::vector<Image> &negative_images, 
    std::vector<Haar_filter> &filters, std::vector<double> &final_classifier_w1, std::vector<double> &final_classifier_w2, 
    std::vector<double> &alfas, std::vector<int> &strong_features_indices, double theta, int rank){
        double lim = 0.0;

    for(int i = 0; i < alfas.size(); i++)
        lim += alfas[i];

    lim *= theta;

    std::cout << "test_negatives" << std::endl;

    for(int i = 0; i < negative_images.size(); i++){
        double f_x = f_of_x(final_classifier_w1, final_classifier_w2, filters, strong_features_indices, negative_images[i], lim);
        if (f_x == 1){
            false_positive[position]++;
        } else if (f_x == -1){
            correct_negative[position]++;
        }
        if (rank == 0) {
            std::cout << "negative_images = " << i << "/" <<  negative_images.size() << 
            "  |  theta = " << theta << std::endl;
        }

    }
    
}

int main (int argc, char* argv[]){

    const int root = 0;
    int rank, world_size;
    
    // Launch MPI processes on each node
    MPI_Init(&argc, &argv);

    // Get the id and number of threads
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::vector<Image> positive_images;
    std::vector<Image> negative_images;
    std::vector<Haar_filter> filters;
    std::vector<double> final_classifier_w1;
    std::vector<double> final_classifier_w2;
    std::vector<double> alfas;
    std::vector<int> strong_features_indices;

    create_filters(filters);
    initialize_images_test(positive_images, "pos");
    initialize_images_test(negative_images, "neg");
    initialize_test_final_classifier_w1(final_classifier_w1);
    initialize_test_final_classifier_w2(final_classifier_w2);
    initialize_test_alfas(alfas);
    initialize_test_strong_features_indices(strong_features_indices);

    // begin of test
    double step = 2.0 / ((double)world_size - 1);
    int vectors_sizes = 2 / step;
    double theta = -1.0 + step*rank;
    double* all_thetas = new double[world_size];

    int correct_positive = 0;
    int* all_correct_positives = new int[world_size];
    int false_positive = 0;
    int* all_false_positives = new int[world_size];
    int correct_negative = 0;
    int* all_correct_negatives = new int[world_size];
    int false_negative = 0;
    int* all_false_negatives = new int [world_size];

    std:: cout << "theta = " << theta << std::endl;

    test_positives(&correct_positive, &false_negative, 0, positive_images, filters, 
        final_classifier_w1, final_classifier_w2, alfas, strong_features_indices, theta, rank);
    test_negatives(&correct_negative, &false_positive, 0, negative_images, filters, 
        final_classifier_w1, final_classifier_w2, alfas, strong_features_indices, theta, rank);

    std::cout << "before gather" << std::endl;

    MPI_Gather(&correct_positive, 1, MPI_INT, all_correct_positives, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Gather(&false_positive, 1, MPI_INT, all_false_positives, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Gather(&correct_negative, 1, MPI_INT, all_correct_negatives, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Gather(&false_negative, 1, MPI_INT, all_false_negatives, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Gather(&theta, 1, MPI_DOUBLE, all_thetas, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

    if (rank == root) {
        ofstream myfile;
        myfile.open ("test_result_small3.txt");
        for (int i = 0; i < world_size; i++){
            double accuracy = 0;
            double total = all_correct_positives[i] + all_correct_negatives[i] + all_false_positives[i] + all_false_negatives[i];
            accuracy = all_correct_positives[i] + all_correct_negatives[i];
            accuracy /= total;
            myfile << "theta = " << all_thetas[i] << ";\n " << "correct_positive = " << all_correct_positives[i] << ";\n "<<
            "false_positive = " << all_false_positives[i] << ";\n " << "correct_negative = " << all_correct_negatives[i] << ";\n " << 
            "false_negative = " << all_false_negatives[i] << ";\n " << "accuracy = " << accuracy << ";\n" << std::endl;
        }
        myfile.close();
    }

    positive_images.clear();
    negative_images.clear();
    filters.clear();
    final_classifier_w1.clear();
    final_classifier_w2.clear();
    alfas.clear();
    strong_features_indices.clear();

    delete[] all_correct_positives;
    delete[] all_correct_negatives;
    delete[] all_false_positives;
    delete[] all_false_negatives;
    delete[] all_thetas;

    MPI_Finalize();
    return 0;
}