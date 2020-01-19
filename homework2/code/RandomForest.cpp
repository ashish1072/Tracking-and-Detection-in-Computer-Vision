#include "RandomForest.h"

RandomForest::RandomForest()
{
}

RandomForest::RandomForest(int treeCount, int maxDepth, int CVFolds, int minSampleCount, int maxCategories)
    :mTreeCount(treeCount), mMaxDepth(maxDepth), mCVFolds(CVFolds), mMinSampleCount(minSampleCount), mMaxCategories(maxCategories)
{
   /*
     construct a forest with given number of trees and initialize all the trees with the
     given parameters
   */
}

RandomForest::~RandomForest()
{
}

void RandomForest::setTreeCount(int treeCount)
{
    // Fill

}

void RandomForest::setMaxDepth(int maxDepth)
{
    mMaxDepth = maxDepth;
    for(uint treeIdx=0; treeIdx<mTreeCount; treeIdx++)
        mTrees[treeIdx]->setMaxDepth(mMaxDepth);
}

void RandomForest::setCVFolds(int cvFols)
{
    // Fill

}

void RandomForest::setMinSampleCount(int minSampleCount)
{
    // Fill
}

void RandomForest::setMaxCategories(int maxCategories)
{
    // Fill
}



void RandomForest::train(/* Fill */)
{
    // Fill
}

float RandomForest::predict(/* Fill */)
{
    // Fill
}

