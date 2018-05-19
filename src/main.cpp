#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.hpp>
#include <opencv2/imgproc.hpp>
#include "../lib/image.hpp"
#include "../lib/haar_filter.hpp"


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


int main () {

    std::string image_name = "../image_test.jpg";
    cv::Mat image = cv::imread(image_name);

    Image *img = new Image(image_name, image);
    cv::imshow("image", img->image);
    cv::waitKey(0);
    cv::imshow("gray image", img->gray_image);
    cv::waitKey(0);
    cv::imshow("integral image", img->integral_image); 
    cv::waitKey(0);

    std::vector<Haar_filter> filters;
    create_filters(img->image, img->integral_image, filters);
    std::cout << "haar filter vector with " << filters.size() << std::endl;

    std::vector<int> features;
    for (int i = 0; i < filters.size(); i++) {
        features.push_back(filters[i].feature(img->integral_image));
    }

    for (int i = 0; i < 10; i++) {
        std::cout << features[i] << std::endl;
    }

    return 0;
}