//
//  super_resolution.cpp
//  SuperResolution
//
//  Created by Hana_Chang on 2015/5/14.
//  Copyright (c) 2015年 Andrea.C. All rights reserved.
//

#include "super_resolution.h"
#include "wavelet_cdf97.h"
#include "interpolator.h"
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using std::vector;


SuperResolution::SuperResolution(Mat &low_resolution_image, const int upsampling_rate, const int total_iteration) {
    this->scaling_factor = 1;
    
    Mat low_resolution_image64;
    low_resolution_image.convertTo(low_resolution_image64, CV_64FC1);
    low_resolution_image64.copyTo(this->low_resolution_image);
    
    this->upsampling_rate = upsampling_rate;
    this->total_iteration = total_iteration;
}

Mat SuperResolution::Run() {
    if (low_resolution_image.channels() == 1) {
        // ----- upsamling scheme -----
        Mat h0Img = usScheme(low_resolution_image);
        
        // ----- gaussian filter -----
        Mat gauImg;
        GaussianBlur(h0Img, gauImg, Size(3,3) , 0.5, 0.5, BORDER_CONSTANT);
        Mat gauImg64;
        gauImg.convertTo(gauImg64, CV_64FC1);
        
        Mat result64 = reconstructIter(0, gauImg64, h0Img);
        Mat result;
        result64.convertTo(result, CV_8UC1);
        
        return result;
        
    } else {
        // ---- rgb2ycbcr ----
        vector<Mat> channels(3);
        split(low_resolution_image, channels);
        Mat B = channels[0];
        Mat G = channels[1];
        Mat R = channels[2];
        
        Mat Y =  0.299*R + 0.587*G + 0.114*B;
        Mat U = -0.168736*R - 0.331264*G + 0.5*B;
        Mat V =  0.5*R - 0.418688*G - 0.081312*B;
        
        // ----- upsamling scheme -----
        Mat h0Img = usScheme(Y);
        
        // ----- gaussian filter -----
        Mat gauImg;
        GaussianBlur(h0Img, gauImg, Size(3,3) , 0.5, 0.5, BORDER_CONSTANT);
        Mat gauImg64;
        gauImg.convertTo(gauImg64, CV_64FC1);
        low_resolution_image = Y;
        Mat resultY = reconstructIter(0, gauImg64, h0Img);
        
        // ---- ycbcr2rgb ----
        Mat upU, upV;
        resize(U, upU, resultY.size(), 0, 0, INTER_LINEAR);
        resize(V, upV, resultY.size(), 0, 0, INTER_LINEAR);
        
        Mat resultR = resultY - 0*upU + 1.4020*upV;
        Mat resultG = resultY - 0.3441*upU - 0.7141*upV;
        Mat resultB = resultY + 1.772*upU + 0*upV;
        
        Mat result64;
        Mat result;
        channels[0] = resultB;
        channels[1] = resultG;
        channels[2] = resultR;
        
        merge(channels, result64);
        result64.convertTo(result, CV_8UC1);
        
        return result;
    }
}

Mat SuperResolution::usScheme(const Mat& img) {

    Mat upImg = Mat::zeros(img.rows*upsampling_rate, img.cols*upsampling_rate, CV_64FC1);
    Interpolator ip(img, 2);
    upImg = ip.BicubicInterpolate();
    
    
    WaveletCdf97 wc(upImg, 1);
    Mat dwtImg = wc.Run();
    
    Mat recoverImg = Mat::zeros( (int) ceil(dwtImg.rows*0.5), (int) ceil(dwtImg.cols*0.5), CV_64FC1);
    resize(img, recoverImg, recoverImg.size(), 0, 0, INTER_AREA);
    
    for (int i = 0; i < recoverImg.rows; i++) {
        for (int j = 0; j < recoverImg.cols; j++) {
            dwtImg.at<double>(i, j) = recoverImg.at<double>(i, j)*scaling_factor;
        }
    }
    
    WaveletCdf97 wc1(dwtImg, -1);
    Mat result = wc1.Run();
    
    return result;
}

Mat SuperResolution::reconstructIter(int current_iteration, const Mat& src, const Mat& hImg) {
    Mat downImg = Mat::zeros(low_resolution_image.rows, low_resolution_image.cols, CV_64FC1);
    
    resize(src, downImg, downImg.size(), 0, 0, INTER_NEAREST);
    Mat downImg64;
    downImg.convertTo(downImg64, CV_64FC1);
    
    Mat reconError = Mat::zeros(downImg64.rows, downImg64.cols, CV_64FC1);
    
    reconError = low_resolution_image - downImg64;
    Mat errImg = usScheme(reconError);
    
    // back-projecting the error
    Mat dst = errImg + hImg;
    if (current_iteration < total_iteration) {
        return reconstructIter(++current_iteration, dst, dst);
    } else {
        return dst;
    }
}