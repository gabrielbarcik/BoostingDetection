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
    int NUMBER_FILES = number_of_files2("/usr/local/INF442-2018/P5/test/" + path);

    std::string image_path;
    cv::Mat image;

    for(int i = 0; i < NUMBER_FILES; i++){
        image_path = "/usr/local/INF442-2018/P5/dev/test/" + path + "/im" + std::to_string(i) + ".jpg";
        image = cv::imread(image_path);
        images.push_back(Image(image_path, image));
    }
}

void initialize_test_final_classifier_w1(std::vector<double> &final_classifier_w1){
    std::ifstream ifile("final_classifier_w1.txt", std::ios::in);

    //check to see that the file was opened correctly:
    if (!ifile.is_open()) {
        std::cerr << "There was a problem opening the input file final_classifier_w1.txt!\n";
        exit(1);//exit or do additional error checking
    }

    double num = 0.0;
    //keep storing values from the text file so long as data exists:
    while (ifile >> num) {
        final_classifier_w1.push_back(num);
    }
}

void initialize_test_final_classifier_w2(std::vector<double> &final_classifier_w2){
    std::ifstream ifile("final_classifier_w2.txt", std::ios::in);

    //check to see that the file was opened correctly:
    if (!ifile.is_open()) {
        std::cerr << "There was a problem opening the input file final_classifier_w2.txt!\n";
        exit(1);//exit or do additional error checking
    }

    double num = 0.0;
    //keep storing values from the text file so long as data exists:
    while (ifile >> num) {
        final_classifier_w2.push_back(num);
    }
}

void initialize_test_alfas(std::vector<double> &alfas){
    std::ifstream ifile("final_alfas.txt", std::ios::in);

    //check to see that the file was opened correctly:
    if (!ifile.is_open()) {
        std::cerr << "There was a problem opening the input file final_alfas.txt!\n";
        exit(1);//exit or do additional error checking
    }

    double num = 0.0;
    //keep storing values from the text file so long as data exists:
    while (ifile >> num) {
        alfas.push_back(num);
    }
}

int f_of_x(std::vector<double> final_classifier_w1, std::vector<double> final_classifier_w2, 
    std::vector<Haar_filter> filters, Image img, int teta){
    double sum = 0;
    for(int i = 0; i < final_classifier_w1.size(); i++){
        sum += filters[i].feature(img.integral_image)*final_classifier_w1[i] + final_classifier_w2[i];
    }

    if(sum >= teta)
        return 1;
    else
        return -1;
}

void test_positives(int* correct_positive, int* false_negative, int position, std::vector<Image> positive_images, 
    std::vector<Haar_filter> filters, std::vector<double> final_classifier_w1, std::vector<double> final_classifier_w2, 
    std::vector<double> alfas, int teta){

    double lim = 0.0;

    for(int i = 0; i < alfas.size(); i++)
        lim += alfas[i];

    lim *= teta;

    for(int i = 0; i < positive_images.size(); i++){
        double f_x = f_of_x(final_classifier_w1, final_classifier_w2, filters, positive_images[i], teta);
        if (f_x == 1){
            correct_positive[position]++;
        } else if (f_x == -1){
            false_negative[position]++;
        }

    }

}

void test_negatives(int* correct_negative, int* false_positive, int position, std::vector<Image> negative_images, 
    std::vector<Haar_filter> filters, std::vector<double> final_classifier_w1, std::vector<double> final_classifier_w2, 
    std::vector<double> alfas, int teta){
        double lim = 0.0;

    for(int i = 0; i < alfas.size(); i++)
        lim += alfas[i];

    lim *= teta;

    for(int i = 0; i < negative_images.size(); i++){
        double f_x = f_of_x(final_classifier_w1, final_classifier_w2, filters, negative_images[i], teta);
        if (f_x == 1){
            false_positive[position]++;
        } else if (f_x == -1){
            correct_negative[position]++;
        }

    }
    
}

int main(){
    std::vector<Image> positive_images;
    std::vector<Image> negative_images;
    std::vector<Haar_filter> filters;
    std::vector<double> final_classifier_w1;
    std::vector<double> final_classifier_w2;
    std::vector<double> alfas;

    create_filters(filters);
    initialize_images_test(positive_images, "pos");
    initialize_images_test(negative_images, "neg");
    initialize_test_final_classifier_w1(final_classifier_w1);
    initialize_test_final_classifier_w2(final_classifier_w2);
    initialize_test_alfas(alfas);

    // begin of test
    double step = 0.05;
    int vectors_sizes = 2 / step;
    double teta = -1.0;

    int* correct_positive = new int [vectors_sizes];
    int* false_positive = new int [vectors_sizes];
    int* correct_negative = new int [vectors_sizes];
    int* false_negative = new int [vectors_sizes];
    double* steps_teta = new double [vectors_sizes];

    for (int i = 0 ; teta <= 1.0; teta += step){
        steps_teta[i] = teta;
        test_positives(correct_positive, false_negative, i, positive_images, filters, final_classifier_w1, final_classifier_w2, alfas, teta);
        test_negatives(correct_negative, false_positive, i, negative_images, filters, final_classifier_w1, final_classifier_w2, alfas, teta);
        i++;
    }

    ofstream myfile;
    myfile.open ("test_result.txt");
    for (int i = 0; i < vectors_sizes; i++){
        myfile << "teta = " << steps_teta[i] << "; " << "correct_positive = " << correct_positive[i] << "; "<<
         "false_positive = " << false_positive[i] << "; " << "correct_negative = " << correct_negative[i] << "; " << 
         "false_negative = " << false_negative[i] << "; " <<  "\n";
    }
    myfile.close();

}