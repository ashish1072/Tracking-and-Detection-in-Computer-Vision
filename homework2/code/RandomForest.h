

#ifndef RF_RANDOMFOREST_H
#define RF_RANDOMFOREST_H


#include <opencv2/opencv.hpp>
#include <vector>

class RandomForest
{
public:
	RandomForest();

    // You can create the forest directly in the constructor or create an empty forest and use the below methods to populate it
	RandomForest(int treeCount, int maxDepth, int CVFolds, int minSampleCount, int maxCategories);
    
    ~RandomForest();

    void setTreeCount(int treeCount);
    void setMaxDepth(int maxDepth);
    void setCVFolds(int cvFols);
    void setMinSampleCount(int minSampleCount);
    void setMaxCategories(int maxCategories);
	

    void train(/* Fill */);

    float predict(/* Fill */);


private:
	int mTreeCount;
	int mMaxDepth;
	int mCVFolds;
	int mMinSampleCount;
	int mMaxCategories;

    // M-Trees for constructing thr forest
    std::vector<cv::Ptr<cv::ml::DTrees> > mTrees;
};

#endif //RF_RANDOMFOREST_H
