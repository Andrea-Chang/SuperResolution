//
//  main.cpp
//  SuperResolution
//
//  Created by Hana_Chang on 2015/5/14.
//  Copyright (c) 2015年 Andrea.C. All rights reserved.
//

#include "super_resolution.h"
#include "interpolator.h"
#include <string>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

int main(int argc, const char * argv[]) {
    int upsample_rate = 2;
    int total_iteration = 2;
    string filename = "/Users/Hana/Desktop/SuperResolution/etc/bin/baboon.jpg";
    Mat origin_img = imread(filename, 1);
    
    SuperResolution SR(origin_img, upsample_rate, total_iteration);
    Mat result = SR.Run();

    imwrite("/Users/Hana/Desktop/SuperResolution/etc/bin/result.png", result);

    return 0;
}
