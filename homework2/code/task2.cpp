#include <opencv2/opencv.hpp>
#include "HOGDescriptor.h"
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>

using namespace cv;
using namespace std;

// Creation of Random Forest.........
void create(Ptr<cv::ml::DTrees> * tree, int num_trees){
int num_class=6;
    for (int i = 0; i < num_trees; i++){
        tree[i] = cv::ml::DTrees::create();
        tree[i]->setCVFolds(0);
        tree[i]->setMaxCategories(num_class);
        tree[i]->setMaxDepth(10);
        tree[i]->setMinSampleCount(2);    
    }
}


// Training Single Tree...........
void train_single_tree(vector<Mat1f> descriptors, Ptr<cv::ml::DTrees> single_tree)
{   int num_class=6;
    Mat1f features;
    Mat image_class;
    std::vector<int> location;
    int num_files[] = {49, 67, 42, 53, 67, 110};
    std::cout << "\nSingle Tree Training...\n\n";
    for (int k = 0; k <= 5; k++){
        for (int j = 0; j < num_files[k]; j++) { 
        location.push_back(j); }
        randShuffle(location);
        for (int i = 0; i < 35; i++){
        features.push_back(descriptors[k].row(location[i]));
        image_class.push_back(k);
        }
        location.clear();
    }
        single_tree->train(features, ml::ROW_SAMPLE, image_class);
}


// Training Random Forest
void train(vector<Mat1f> descriptors, Ptr<cv::ml::DTrees> * tree, int num_trees)
{   int num_files[] = {49, 67, 42, 53, 67, 110};
    std::cout << "\n\nRandom Forest Training.....\n";
    std::vector<int> location;
    vector<Mat> image_class(num_trees);
    vector<Mat1f> features(num_trees);
    for (int t = 0; t < num_trees; ++t){
        for (int k = 0; k < 6; k++){
            for (int j = 0; j < num_files[k]; j++) { 
            location.push_back(j); }
            randShuffle(location);
            for (int i = 0; i < int(num_files[k]*.60); i++){
            features[t].push_back(descriptors[k].row(location[i]));
            image_class[t].push_back(k);
            }
            location.clear();
        }
    }
    for (int t = 0; t < num_trees; t++) {
        tree[t]->train(features[t],ml::ROW_SAMPLE, image_class[t]);}
}


// Prediciting Classification from Single Tree
void predict_single_tree(Ptr<cv::ml::DTrees> single_tree){
String result, imageName;
Size win_size = cv::Size(96, 96);
Size block_size = cv::Size(24,24);
Size block_step = cv::Size(24,24);
Size cell_size = cv::Size(12,12);
Size pad_size = cv::Size(0,0);
Size sz = cv::Size(96, 96);
int nbins = 9;
int num_files[] = {49, 67, 42, 53, 67, 110};
double label;
float avg=0;
    for (int i = 0; i < 6; i++){ 
        int match = 0;
        for (int j = 0; j < 10; j++){
            stringstream imageP;
            imageP << string("/home/ashish/Documents/CSE/3rd sem/TDCV/tut/homework 2/data/task2/test/") << setfill('0') << setw(2) << i << "/" << setfill('0') << setw(4) << j + num_files[i] << ".jpg";
            string imgFile = imageP.str();
            vector<float> descriptors;
            cv::Mat image = cv::imread(imgFile);
            cv::resize(image, image, sz);
            //cv::cvtColor(image, image, CV_RGB2GRAY );
            cv::HOGDescriptor hog( win_size, block_size, block_step, cell_size, nbins);
            hog.compute(image, descriptors, block_step, pad_size);
            Mat1f descriptor_data(1, descriptors.size(), descriptors.data());
            label = single_tree->predict(descriptor_data);
        
            if (int(label) == i) 
            { match++ ;}
        }
        avg = avg + match;
        std::cout << "Accuracy for Class " << i <<" =  " << (match/10.0)*100<<"%" <<std::endl;
    }
        std::cout << "Average accuracy of single tree: " << (avg/6/10)*100 <<"%" <<std::endl;
}


// Prediciting Classification from Random Forest
void predict(Ptr<cv::ml::DTrees> * tree, int num_trees){
Size win_size = cv::Size(96, 96);
Size block_size = cv::Size(24,24);
Size block_step = cv::Size(24,24);
Size cell_size = cv::Size(12,12);
Size pad_size = cv::Size(0,0);
Size sz = cv::Size(96, 96);
int nbins = 9;
String imageName;
int num_files[] = {49, 67, 42, 53, 67, 110};
float avg =0;
    
    for (int i = 0; i < 6; i++){ 
        std::cout << "\nResults for class " << i << ":\n";
        float match = 0; float mismatch = 0; float Total_confidence = 0;

        for (int j = 0; j < 10; j++){
            stringstream imageP;
            imageP << string("/home/ashish/Documents/CSE/3rd sem/TDCV/tut/homework 2/data/task2/test/") << setfill('0') << setw(2) << i << "/" << setfill('0') << setw(4) << j + num_files[i] << ".jpg";
            string imgFile = imageP.str();

            vector<float> descriptors;
            cv::Mat image = cv::imread(imgFile);
            cv::resize(image, image, sz);
            cv::cvtColor(image, image, CV_RGB2GRAY );
            cv::HOGDescriptor hog( win_size, block_size, block_step, cell_size, nbins);
            hog.compute(image, descriptors, block_step, pad_size);

            Mat1f descriptor_data(1, descriptors.size(), descriptors.data());
            vector<int> prediction = {0,0,0,0,0,0};

            for (int t = 0; t < num_trees; t++){ 
                double label = tree[t]->predict(descriptor_data);
                prediction.at((int)label) = prediction.at((int)label) + 1;
            }
            int maxElementIndex = std::max_element(prediction.begin(),prediction.end()) - prediction.begin();
            if (maxElementIndex == i) { match++; }
            else mismatch++;
        }
        float accuracy = (match /(match + mismatch));
        accuracy = accuracy*100;
        avg = avg + accuracy;
        std::cout << "Accuracy: " << accuracy <<"%" <<std::endl;
    }
    std::cout << "Average accuracy of random forest: " << (avg/6) <<"%" <<std::endl;
}



int main()
{
Size win_size = cv::Size(96, 96);
Size block_size = cv::Size(24,24);
Size block_step = cv::Size(24,24);
Size cell_size = cv::Size(12,12);
Size pad_size = cv::Size(0,0);
Size sz = cv::Size(96, 96);
int nbins = 9;
int num_trees = 60;
int num_class=6;
vector<Mat1f> descriptors(num_class);
Mat labels;
std::vector<float> computed_descriptors;
int num_files[] = {49, 67, 42, 53, 67, 110};

    for (int i = 0; i < num_class; i++)
    { for (int j = 0; j < num_files[i]; j++)
        {   stringstream imageP;
            imageP << string("/home/ashish/Documents/CSE/3rd sem/TDCV/tut/homework 2/data/task2/train/") << setfill('0') << setw(2) << i << "/" << setfill('0') << setw(4) << j << ".jpg";
            string imgFile = imageP.str();

            cv::Mat image = cv::imread(imgFile, IMREAD_COLOR);
             
            cv::resize(image, image, sz);
            cv::cvtColor(image, image, CV_RGB2GRAY );
            cv::HOGDescriptor hog( win_size, block_size, block_step, cell_size, nbins);
            hog.compute(image, computed_descriptors, block_step, pad_size);
            Mat1f descriptor_data(1, computed_descriptors.size(), computed_descriptors.data());
            descriptors[i].push_back(descriptor_data);
           // cout<<"After 3rd loop"<<endl;
            labels.push_back(i);
        }
    }

    // cout<<"Descriptors created"<<endl;
    Ptr<cv::ml::DTrees> tree[num_trees];
    create(tree, num_trees);
    Ptr<cv::ml::DTrees> single_tree = cv::ml::DTrees::create();
    single_tree->setCVFolds(0);
    single_tree->setMaxCategories(num_class);
    single_tree->setMaxDepth(10);
    single_tree->setMinSampleCount(2); 
    train_single_tree(descriptors, single_tree);
    predict_single_tree(single_tree);
    train(descriptors, tree, num_trees);
    predict(tree, num_trees);
    cout<<"\n>> Program End"<<endl;

    return 0;
}











