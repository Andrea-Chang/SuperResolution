//
//  super_resolution.h
//  SuperResolution
//
//  Created by Hana_Chang on 2015/5/14.
//  Copyright (c) 2015年 Andrea.C. All rights reserved.
//

#ifndef SUPER_RESOLUTION_H_
#define SUPER_RESOLUTION_H_
#include <opencv2/opencv.hpp>

namespace cv {
    class Mat;
}

class SuperResolution {
private:
    int scaling_factor;
    
    cv::Mat usScheme(const cv::Mat& img);
    cv::Mat reconstructIter(int current_iteration, const cv::Mat& src, const cv::Mat& hImg);
    cv::Mat low_resolution_image;
    
public:
    int upsampling_rate;
    int total_iteration;
    
    SuperResolution(cv::Mat &low_resolution_image, const int upsampling_rate, int total_iteration);
    cv::Mat Run();
};


#endif /* SUPER_RESOLUTION_H_ */