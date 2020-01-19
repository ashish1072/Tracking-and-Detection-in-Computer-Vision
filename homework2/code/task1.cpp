
#include <opencv2/opencv.hpp>
#include "HOGDescriptor.h"
#include <iostream>


using namespace std;
//using namespace cv;


int main(){

    // Reading and displaying the original image
    cv::Mat image = cv::imread("/home/ashish/Documents/CSE/3rd sem/TDCV/tut/homework 2/data/task1/obj1000.jpg", cv::IMREAD_COLOR);
    cv::namedWindow("original image", CV_WINDOW_AUTOSIZE);
	cv::imshow("original image", image);

    // Gray-scale conversion and display
    cv::Mat gray_image;
    cv::cvtColor( image, gray_image, CV_RGB2GRAY );
    cv::namedWindow("gray_image", CV_WINDOW_AUTOSIZE);
	cv::imshow("gray_image", gray_image);

    // Resizing: Compression and display
    cv::Mat compressed_image;
    cv::resize(image, compressed_image, cv::Size(), 0.5, 0.5);
    cv::namedWindow("compressed_image", CV_WINDOW_AUTOSIZE);
	cv::imshow("compressed_image", compressed_image);

    // Resizing: Expansion and display
    cv::Mat enlarged_image;
    cv::resize(image, enlarged_image, cv::Size(), 2.0, 2.0);
    cv::namedWindow("enlarged_image", CV_WINDOW_AUTOSIZE);
	cv::imshow("enlarged_image", enlarged_image);

    // Rotation and display
    cv::Mat rotated_image;
    cv::rotate(image, rotated_image, cv::ROTATE_90_COUNTERCLOCKWISE);
    cv::namedWindow("rotated_image", CV_WINDOW_AUTOSIZE);
	cv::imshow("rotated_image", rotated_image);

    // Flipping and display
    cv::Mat flipped_image;
    cv::flip(image, flipped_image, -1);
    cv::namedWindow("flipped_image", CV_WINDOW_AUTOSIZE);
	cv::imshow("flipped_image", flipped_image);
    //cv::waitKey(0);
	//Fill Code here
    /*
    	* Create instance of HOGDescriptor and initialize
    	* Compute HOG descriptors
    	* visualize
    */
   
	HOGDescriptor hog_detector;
	vector<float> feat;
	hog_detector.initDetector();
	hog_detector.detectHOGDescriptor(image, feat);
    return 0;
}