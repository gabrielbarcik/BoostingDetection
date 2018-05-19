#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.hpp>
#include <opencv2/imgproc.hpp>
#include "../lib/image.hpp"

Image::Image(std::string name, cv::Mat image) {
    this->name = name;
    this->image = image; 
    cvtColor(this->image, this->gray_image, CV_RGB2GRAY);
    this->integral_image = cv::Mat::zeros(gray_image.rows, gray_image.cols, CV_32F);
    create_integral_image();
}

void Image::create_integral_image() {
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
