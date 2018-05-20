#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.hpp>
#include <opencv2/imgproc.hpp>
#include "../lib/image.hpp"
#include "../lib/haar_filter.hpp"
#include "../lib/classifier.hpp"

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

void create_filters (cv::Mat &image, cv::Mat &integral_image, std::vector<Haar_filter> &filters) {
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
        while (ep = readdir (dp)){
          num_files++;
          std::cout << "file name: " << ep->d_name << std::endl;
        }

        (void) closedir (dp);
    } else
        perror ("Couldn't open the directory");

    return num_files;
}


/* funcion used in calculate_features to find randomly a image in the learning base and calculate
    it`s features and modify ck. ck = 1 if the image has a visage and -1 if not.
*/
Image find_random_image_learning (int &ck, std::vector<int> features){
    double r = ((double) rand() / (RAND_MAX));
    std::string image_path = "";

    if(r < 0.5){
        ck = -1;
        std::string image_path = "neg/";
        srand((unsigned)time(NULL));
        r = (rand()%(number_of_files(image_path)-1)) + 1;

        image_path = "neg/im" + std::to_string(r) + ".jpg";
        
    } else {
        ck = 1;
        std::string image_path = "pos/";
        r = (rand()%(number_of_files(image_path)-1)) + 1;
        image_path = "pos/im" + std::to_string(r) + ".jpg";
    }
    // TODO: pegar as features da imagem
    //features = 

    cv::Mat image = cv::imread(image_path);

    Image img = Image(image_path, image);

    return img;
}

void calculate_features(std::vector<Classifier> &classifiers){
    int K = 4; // TODO: evaluate the right value for K
    double epsilon = 0.005; // TODO: evaluate the right value for epsilon
    std::vector<int> features;

    for(int k = 1, ck; k <= K; k++){
        
        Image imgage = find_random_image_learning(ck, features); 
                                                            
        classifiers = std::vector<Classifier>(features.size());
        for(unsigned long i = 0, Xki, h; i < features.size(); i++){
            
            Xki = features[i]; 
            if (classifiers[i].w1 * Xki + classifiers[i].w2 >= 0)
                h = 1;
            else h = -1;
            classifiers[i].w1 -= epsilon * (h - ck) * Xki;
            classifiers[i].w2 -= epsilon * (h - ck);
        }
        features.clear();
    }

    return classifiers;
}



int main () {
    std::vector<Classifier> classifiers;

    calculate_features(classifiers);

    return 0;
}