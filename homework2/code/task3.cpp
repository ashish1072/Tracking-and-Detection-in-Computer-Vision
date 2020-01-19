#include <opencv2/opencv.hpp>
#include "HOGDescriptor.h"
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>
#include "nms.hpp"
#include "utils.hpp"

using namespace cv;
using namespace std;
using cv::Rect;
using cv::Point;


int num_files[] = {400, 620, 400, 2300};
int num_trees = 30;
float prob_threshold[] = {0.60,0.75,0.80};
float nms_thresh = 0.01;
int stride = 5;
std::vector<cv::Size> win_sizes = {cv::Size(80,80), cv::Size(110,110), cv::Size(130,130), cv::Size(150,150)};
int tot_classes = 4;


// intersection over unioin
int iou(std::vector<float> gt, cv::Rect r2, float threshold)
{   
    cv::Rect2d rect_1(gt[0], gt[1], gt[2] - gt[0], gt[3] - gt[1]);
    cv::Rect2d rect_2(r2.x, r2.y, r2.width, r2.height);
    cv::Rect2d r3 = rect_1 & rect_2;
    float overlap;
    if (r3.area() > 0){ overlap = r3.area()/ (rect_1.area()+rect_2.area()- r3.area());}
    else overlap = 0;

    if(overlap > threshold) return 1;
    else return 0;
}

// Computation of Precision and Recall
void precision_and_recall(float & precision, float & recall, std::vector<cv::Rect> & red_rect_0, 
std::vector<cv::Rect> & red_rect_1, std::vector<cv::Rect> & red_rect_2, std::string image_num)
{
    std::vector< std::vector<float> > coordinates_gt;
    std::string line;
    std::fstream my_file(image_num.c_str(), std::ios::in);

    if (my_file.is_open()){ 
            int count = 0;
            while(count < 3){
            getline(my_file, line);
            std::istringstream iss(line);
            std::vector<std::string> coordinates(std::istream_iterator<std::string>{iss},
                                                std::istream_iterator<std::string>());
            count++;
            coordinates_gt.push_back(std::vector<float>{std::stof(coordinates[1]), std::stof(coordinates[2]),
                                                        std::stof(coordinates[3]), std::stof(coordinates[4])});
        }
    }

    // predictions for class 0
    int red_correct_0 = 0;
    int red_total_0 = 0;
    for (int i = 0; i < red_rect_0.size(); i++){
    int iou_val = iou(coordinates_gt.at(0), red_rect_0.at(i), 0.5);
        if (iou_val == 1){
        red_correct_0++;}
        red_total_0++;
    }

    // predictions for class 1
    int red_correct_1 = 0;
    int red_total_1 = 0;
    for (int i = 0; i < red_rect_1.size(); i++){
    int iou_val = iou(coordinates_gt.at(1), red_rect_1.at(i), 0.5);
        if (iou_val == 1){
        red_correct_1++;}
        red_total_1++;
    }

    // predictions for class 2
    int red_correct_2 = 0;
    int red_total_2 = 0;
    for (int i = 0; i < red_rect_2.size(); i++){
    int iou_val = iou(coordinates_gt.at(2), red_rect_2.at(i), 0.5);
    if (iou_val == 1){
    red_correct_2++;}
    red_total_2++;
    }

    precision = (red_correct_0 + red_correct_1 + red_correct_2) / (float)(red_total_0 + red_total_1 + red_total_2);
    recall = (red_correct_0 + red_correct_1 + red_correct_2) / (float)3;
}

// Calculating Hog Descriptor
void calc_hog(Mat image, vector<float>& computed_descriptors){
    cv::resize(image, image, Size(168, 168));
    //cv::cvtColor(image, image, CV_RGB2GRAY );
    cv::HOGDescriptor hog(Size(168, 168), Size(24, 24), Size(24, 24), Size(12, 12), 9);
    hog.compute(image, computed_descriptors, Size(0, 0), Size(0, 0));
}

// Creating Random Forest
void create(Ptr<cv::ml::DTrees> * tree, int num_trees){
for (int idx = 0; idx < num_trees; idx++){
    tree[idx] = cv::ml::DTrees::create();
    tree[idx]->setMaxDepth(20);
    tree[idx]->setMinSampleCount(5);
    tree[idx]->setCVFolds(0);
    tree[idx]->setMaxCategories(tot_classes);
   }
}

// Training Random Forest
void train(vector<Mat1f> descriptors, Ptr<cv::ml::DTrees> * tree, int num_trees)
{   std::cout << "\n\nRandom Forest Training.....\n";
    std::vector<int> location;
    vector<Mat> image_class(num_trees);
    vector<Mat1f> features(num_trees);
    for (int t = 0; t < num_trees; ++t){
        for (int k = 0; k < 4; k++){
            for (int j = 0; j < num_files[k]; j++) { 
            location.push_back(j); }
            randShuffle(location);
            for (int i = 0; i < int(num_files[k]*.6); i++){
            features[t].push_back(descriptors[k].row(location[i]));
            image_class[t].push_back(k);
            }
            location.clear();
        }
    }
    for (int t = 0; t < num_trees; t++) {
        std::cout << "Tree No: " << t <<std::endl;
        tree[t]->train(features[t],ml::ROW_SAMPLE, image_class[t]);}
}

//Predicting the classification
void predict(Ptr<cv::ml::DTrees> * tree, int num_trees, Mat& test_features, 
            std::vector< std::vector<float> > & final_locations, 
            std::vector< std::vector<float> > & locations)
{   for (int j = 0; j < test_features.rows; j++)
    {
        float curr;
        Mat1f waste_array;

        int predictd_class[4]= {0,0,0,0};

        for (int tree_idx = 0; tree_idx < num_trees; tree_idx++){
            curr = tree[tree_idx]->predict(test_features.row(j), waste_array);
            predictd_class[(int)curr]++;
        }

        for (int i = 0;i< 3; ++i){
        if ( predictd_class[i]/(float)num_trees > prob_threshold[i]){
            std::vector<float> temp = {(float)i, locations.at(j).at(0), locations.at(j).at(1), locations.at(j).at(2), locations.at(j).at(3)};
            final_locations.push_back(temp);
           }
        }
    }
}

// Creating windows
int get_final_locations(std::string file_path, std::vector< std::vector<float> > & final_locations, Ptr<cv::ml::DTrees> * tree, int num_trees)
{
    cv::Mat image, crop;
	cv::Rect roi;
    image = cv::imread(file_path, IMREAD_GRAYSCALE);
    int count = 0, count_refined = 0;
    for (int win = 0; win < win_sizes.size(); win++){
    cv::Mat test_data;
    std::vector< std::vector<float> > locations;
    cv::Size s = win_sizes[win];
    int col_length = (image.rows - s.height)/stride + 1;
    int row_length = (image.cols - s.width)/stride + 1;
    count = 0;

    for (int r = 0; r < row_length; r++){
        for (int col = 0; col < col_length; col++){
        roi = cv::Rect(r * stride, col * stride, s.width, s.height);
        image(roi).copyTo(crop);
        std::vector<float> descriptor;
        calc_hog(crop, descriptor);
        Mat1f row(1, descriptor.size(), descriptor.data());
        test_data.push_back(row);
        std::vector<float> temp = {(float)r * stride, (float)col * stride, (float)s.width, (float)s.height};
        locations.push_back(temp);
        descriptor.clear();
            }
       }
        predict(tree, num_trees, test_data, final_locations, locations);
        locations.clear();
    }

}

int main(int argc, char** argv){

Size win_size = cv::Size(168, 168);
Size block_size = cv::Size(24,24);
Size block_step = cv::Size(24,24);
Size cell_size = cv::Size(12,12);
Size pad_size = cv::Size(0,0);
Size sz = cv::Size(168, 168);
int nbins = 9;
//int num_trees = 60;
int num_class=4;
vector<Mat1f> descriptors(num_class);
Mat labels;
std::vector<float> computed_descriptors;
  
 cout<<"\nCalculating hog descriptors for training set..."<<endl;
 
    for (int i = 0; i < num_class; i++)
    { for (int j = 0; j < num_files[i]; j++)
        {   stringstream imageP;
            imageP << string("/home/ashish/Documents/CSE/3rd sem/TDCV/tut/homework 2/data/task3/train_aug/") << setfill('0') << setw(2) << i << "/" << j << ".jpg";
            string imgFile = imageP.str();

            cv::Mat image = cv::imread(imgFile, IMREAD_COLOR);
             
            cv::resize(image, image, sz);
            cv::cvtColor(image, image, CV_RGB2GRAY );
            cv::HOGDescriptor hog( win_size, block_size, block_step, cell_size, nbins);
            hog.compute(image, computed_descriptors, Size(0,0), pad_size);
            Mat1f descriptor_data(1, computed_descriptors.size(), computed_descriptors.data());
            descriptors[i].push_back(descriptor_data);
           // cout<<"After 3rd loop"<<endl;
            labels.push_back(i);
        }
    }

    // cout<<"Descriptors created"<<endl;
    Ptr<cv::ml::DTrees> tree[num_trees];
    cout<<"\nCreating random forest...";
    create(tree, num_trees);
    train(descriptors, tree, num_trees);

    std::vector< std::vector<float> > final_locations;
    std::string image_path = std::string("/home/ashish/Documents/CSE/3rd sem/TDCV/tut/homework 2/data/task3/test/") + std::string(argv[1]) +std::string(".jpg");
    std::string file_path = std::string("/home/ashish/Documents/CSE/3rd sem/TDCV/tut/homework 2/data/task3/gt/") + std::string(argv[1]) +std::string(".gt.txt");

    get_final_locations(image_path, final_locations, tree, num_trees);
    
    std::vector< std::vector<float> > rect_0, rect_1, rect_2;
    
    for (int i = 0; i < final_locations.size(); i++)
    {
        std::vector<float> temp = { final_locations.at(i).at(1), 
                                    final_locations.at(i).at(2), 
                                    final_locations.at(i).at(1) + final_locations.at(i).at(3), 
                                    final_locations.at(i).at(2) + final_locations.at(i).at(4)};

        switch (int(final_locations.at(i).at(0)))
        {
            case 0:
                rect_0.push_back(temp);
                break;
            case 1:
                rect_1.push_back(temp);
                break;
            case 2:
                rect_2.push_back(temp);
                break;
        }
    }
    
    std::vector<cv::Rect> reducedRectangle_0 = nms(rect_0, nms_thresh);
    std::vector<cv::Rect> reducedRectangle_1 = nms(rect_1, nms_thresh);
    std::vector<cv::Rect> reducedRectangle_2 = nms(rect_2, nms_thresh);

    cv::Mat image = cv::imread(image_path);
    
    cv::namedWindow( "label - 0", cv::WINDOW_AUTOSIZE );
    cv::namedWindow( "label - 1", cv::WINDOW_AUTOSIZE );
    cv::namedWindow( "label - 2", cv::WINDOW_AUTOSIZE );
    
    Mat image_clone_1, image_clone_2;
    image_clone_1 = image.clone();
    image_clone_2 = image.clone();

    DrawRectangles(image, reducedRectangle_0);
    DrawRectangles(image_clone_1, reducedRectangle_1);
    DrawRectangles(image_clone_2, reducedRectangle_2);
    
    cv::imshow("label - 0", image);
    cv::imshow("label - 1", image_clone_1);
    cv::imshow("label - 2", image_clone_2);
    
    float precision = 0, recall = 0;

    precision_and_recall(precision, recall, reducedRectangle_0, reducedRectangle_1, reducedRectangle_2, file_path);
    std::cout << "\nPrecision: " << precision;
    std::cout << "\nRecall: " << recall << std::endl;
    cv::waitKey(0);
    return 0;
}