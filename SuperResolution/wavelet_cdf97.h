//
//  wavelet_cdf97.h
//  SuperResolution
//
//  Created by Hana_Chang on 2015/5/14.
//  Copyright (c) 2015年 Andrea.C. All rights reserved.
//

#ifndef WAVELET_CDF97_H_
#define WAVELET_CDF97_H_
#include <opencv2/opencv.hpp>
#include <vector>

namespace cv {
    class Mat;
}

class WaveletCdf97 {
private:
    friend class SuperResolution;
    struct WaveletParam {
        cv::Mat lift_filter;
        std::vector<double> extrapolate_odd;
        double scale_factor;
        int enable;
        WaveletParam() {
            enable = 0 ;
        };
        
    };
    
    cv::Mat wavelet_img;
    int nlevel;
    WaveletParam wp;
    
    void ForwardTransform();
    void InverseTransform();
    void TransferAlongRow(const int N1, const int N2, const int M1, const int M2, const bool flag);
    void TransferAlongCol(const int N1, const int N2, const int M1, const int M2, const bool flag);
    cv::Mat filter(const cv::Mat& B, const cv::Mat& X, const cv::Mat& Zi); // b is a column vector
    
public:
    WaveletCdf97(cv::Mat& img, const int nlevel, WaveletParam w = WaveletParam());
    cv::Mat Run();
};


#endif /* WAVELET_CDF97_H_ */