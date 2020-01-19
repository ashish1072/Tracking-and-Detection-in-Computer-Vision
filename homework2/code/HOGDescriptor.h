#ifndef RF_HOGDESCRIPTOR_H
#define RF_HOGDESCRIPTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>


class HOGDescriptor {


    private:
        cv::Size win_size;

        /*
            Fill other parameters here
        */
        cv::Size block_size;
        cv::Size block_step;
        cv::Size cell_size;
        cv::Size pad_size;
        int nbins;

        cv::HOGDescriptor hog_detector;

    public:
        cv::HOGDescriptor getHog_detector();

    private:
        bool is_init;

    public:

    HOGDescriptor() {
        //initialize default parameters(win_size, block_size, block_step,....)
        win_size = cv::Size(128, 128);

        //Fill other parameters here
        block_size = cv::Size(16,16);

        block_step = cv::Size(8,8);

        cell_size = cv::Size(8,8);

        pad_size = cv::Size(0,0);

        nbins = 9;
  
        // parameter to check if descriptor is already initialized or not
        is_init = false;
    };


    void setWinSize(cv::Size sz){
        //Fill
        win_size = sz;    
    }

    cv::Size getWinSize(){
        //Fill
        return win_size;
    }

    void setBlockSize(cv::Size sz) {
        //Fill
        block_size = sz;
    }

    void setBlockStep(cv::Size sz) {
       //Fill
        block_step = sz;
    }

    cv::Size getBlockStep(){
        //Fill
        return block_step;
    }

    void setCellSize(cv::Size sz) {
      //Fill
        cell_size = sz;
    }

    void setPadSize(cv::Size sz) {
        pad_size = sz;
    }

    cv::Size getPadSize(){
        //Fill
        return pad_size;
    }


    void initDetector();

    void visualizeHOG(cv::Mat img, std::vector<float> &feats, cv::HOGDescriptor hog_detector, int scale_factor);

    void detectHOGDescriptor(cv::Mat &im, std::vector<float> &feat);

    ~HOGDescriptor() {};

};

#endif //RF_HOGDESCRIPTOR_H
