#ifndef IMAGEH
#define IMAGEH

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.hpp>
#include <opencv2/imgproc.hpp>

class Image {
    public:
        Image(std::string name, cv::Mat image);
        cv::Mat image;
        cv::Mat gray_image;
        cv::Mat integral_image;

    private:
        // variables
        std::string name;
        std::vector<int> haar_features;

        // methods
        void create_integral_image();
};

#endif