//
//  wavelet_cdf97.cpp
//  SuperResolution
//
//  Created by Hana_Chang on 2015/5/14.
//  Copyright (c) 2015年 Andrea.C. All rights reserved.
//

#include "wavelet_cdf97.h"

using cv::Mat;
using std::vector;

WaveletCdf97::WaveletCdf97(Mat& img, const int nlevel, WaveletParam w) {
    // input image initialization
    this->wavelet_img = img;
    this->nlevel = nlevel;
    
    // Wavelet cdf 9/7 parameters setting
    // 1. lift filter
    if (w.enable == 0) {
        double lift_filter_arr[8] = {-1.5861343420693648, -0.0529801185718856, 0.8829110755411875, 0.4435068520511142, -1.5861343420693648, -0.0529801185718856, 0.8829110755411875, 0.4435068520511142};
        w.scale_factor = 1.1496043988602418;
        
        double S1 = lift_filter_arr[0];
        double S2 = lift_filter_arr[1];
        double S3 = lift_filter_arr[2];
        Mat tmp(2, 4, CV_64FC1, lift_filter_arr);
        
        w.lift_filter = tmp;
        
        // 2. extrapolate
        w.extrapolate_odd.push_back( -2*S1*S2*S3/(1+2*S2*S3) );
        w.extrapolate_odd.push_back( -2*S2*S3/(1+2*S2*S3) );
        w.extrapolate_odd.push_back( -2*(S1+S3+3*S1*S2*S3)/(1+2*S2*S3) );
        this->wp.scale_factor = w.scale_factor;
        this->wp.extrapolate_odd = w.extrapolate_odd;
        w.lift_filter.copyTo(wp.lift_filter);
        
    } else {
        this->wp.extrapolate_odd = w.extrapolate_odd;
        w.lift_filter.copyTo(this->wp.lift_filter);
        this->wp.scale_factor = w.scale_factor;
        this->wp = w;
    }
}

Mat WaveletCdf97::Run() {
    if (nlevel >= 0) {
        ForwardTransform();
    } else {
        InverseTransform();
    }
    
    return wavelet_img;
}

void WaveletCdf97::ForwardTransform() {
    int N1 = wavelet_img.rows;
    int N2 = wavelet_img.cols;
    int M1 = 0;
    int M2 = 0;
    
    //---- forward transform ----
    if (nlevel >= 0) {
        for (int k = 0; k < nlevel; k++) {
            M1 = (int) ceil((double)(N1/2.0));
            M2 = (int) ceil((double)(N2/2.0));
            // transform along columns
            if (N1 > 1) {
                TransferAlongCol(N1, N2, M1, M2, 0);
            }
            // transform along rows
            if (N2 > 1) {
                TransferAlongRow(N1, N2, M1, M2, 0);
            }
            N1 = M1;
            N2 = M2;
        }
    }
}

void WaveletCdf97::InverseTransform() {
    int N1 = wavelet_img.rows;
    int N2 = wavelet_img.cols;
    int M1 = 0;
    int M2 = 0;
    for (int k = 1+nlevel; k < 1; k++) {
        M1 = (int) ceil( N1*pow(2.0, (double)k) );
        M2 = (int) ceil( N2*pow(2.0, (double)k) );
        // transform along rows
        if (M2 > 1) {
            TransferAlongRow(N1, N2, M1, M2, 1);
        }
        // transform along cols
        if (M1 > 1) {
            TransferAlongCol(N1, N2, M1, M2, 1);
        }
    }
}

void WaveletCdf97::TransferAlongCol(const int N1, const int N2, const int M1, const int M2, const bool flag) {
    // for forward transform
    if (flag == 0) {
        Mat X0 = Mat::zeros((N1-1)/2+1, N2, CV_64FC1);
        Mat X1 = Mat::zeros((N1-1)/2+1, N2, CV_64FC1);
        vector<int> rightShiftIdx;
        for (int i = 1; i < M1; ++i) {
            rightShiftIdx.push_back(i);
        }
        
        rightShiftIdx.push_back(M1-1);
        
        
        for (int i = 0; i < X0.rows; i++) {
            for (int j = 0; j < X0.cols; j++) {
                X0.at<double>(i, j) = wavelet_img.at<double>(i*2, j);
            }
        }
        
        // apply lifting stages
        Mat X0_rightShift = Mat::zeros(rightShiftIdx.size(), X0.cols, CV_64FC1); // X0(rightShift,:,:)
        for (int i = 0; i < X0_rightShift.rows; i++) {
            for (int j = 0; j < X0_rightShift.cols; j++) {
                X0_rightShift.at<double>(i, j) = X0.at<double>(rightShiftIdx[i], j);
            }
        }
        
        if (N1 % 2 == 1) {
            Mat tmp = filter(wp.lift_filter.col(0), X0_rightShift, X0.row(0)*wp.lift_filter.at<double>(0,0));
            for (int j = 0; j < X1.cols; j++) {
                for (int i = 0; i < X1.rows-1; i++) {
                    X1.at<double>(i, j) = wavelet_img.at<double>(i*2+1, j);
                }
                X1.at<double>(X1.rows-1, j) = X0.at<double>(M1-2, j)*wp.extrapolate_odd[0]
                + wavelet_img.at<double>(N1-2, j)*wp.extrapolate_odd[1]
                + X0.at<double>(M1-1, j)*wp.extrapolate_odd[2];
            }
            X1 = X1 + tmp;
            
        } else {
            
            Mat tmp = filter(wp.lift_filter.col(0), X0_rightShift, X0.row(0)*wp.lift_filter.at<double>(0,0));
            for (int i = 0; i < X1.rows; i++) {
                for (int j = 0; j < X1.cols; j++) {
                    X1.at<double>(i, j) = wavelet_img.at<double>(i*2+1, j);
                }
            }
            X1 = X1 + tmp;
            
        }
        
        X0 = X0 + filter(wp.lift_filter.col(1), X1, X1.row(0)*wp.lift_filter.at<double>(0,1));
        
        for (int i = 0; i < X0_rightShift.rows; i++) {
            for (int j = 0; j < X0_rightShift.cols; j++) {
                X0_rightShift.at<double>(i, j) = X0.at<double>(rightShiftIdx[i], j);
            }
        }
        
        X1 = X1 + filter(wp.lift_filter.col(2), X0_rightShift, X0.row(0)*wp.lift_filter.at<double>(0,2));
        
        X0 = X0 + filter(wp.lift_filter.col(3), X1, X1.row(0)*wp.lift_filter.at<double>(0,3));
        
        Mat new_X1;
        if (N1 % 2 == 1) {
            new_X1 = Mat::zeros(X1.rows-1, X1.cols, CV_64FC1);
            for (int i = 0; i < new_X1.rows; i++) {
                for (int j = 0; j < new_X1.cols; j++) {
                    new_X1.at<double>(i, j) = X1.at<double>(i, j);
                }
            }
            
        } else {
            new_X1 = X1;
        }

        for (int i = 0; i < X0.rows; i++) {
            for (int j = 0; j < X0.cols; j++) {
                wavelet_img.at<double>(i, j) = X0.at<double>(i, j)*wp.scale_factor;
            }
        }
        
        for (int i = 0; i < new_X1.rows; i++) {
            for (int j = 0; j < new_X1.cols; j++) {
                wavelet_img.at<double>(i+X0.rows, j) = new_X1.at<double>(i, j)/wp.scale_factor;
            }
        }
        
        // for inverse transform
    } else {
        int Q = (int) ceil((double)(M1/2.0));
        Mat X0 = Mat::zeros(Q, M2, CV_64FC1);
        Mat X1 = Mat::zeros(M1-Q, M2, CV_64FC1);
        vector<int> rightShiftIdx;
        for (int i = 1; i < Q; ++i) {
            rightShiftIdx.push_back(i);
        } rightShiftIdx.push_back(Q-1);
        
        for (int i = 0; i < X1.rows; i++) {
            for (int j = 0; j < X1.cols; j++) {
                X1.at<double>(i, j) = wavelet_img.at<double>(i+Q, j)*wp.scale_factor;
            }
        }
        
        Mat add_X1;
        if (M1 % 2 == 1) {
            add_X1 = Mat::zeros(X1.rows+1, X1.cols, CV_64FC1);
            for (int i = 0; i < X1.rows; i++) {
                for (int j = 0; j < X1.cols; j++) {
                    add_X1.at<double>(i, j) = X1.at<double>(i, j);
                }
            }
            
        } else {
            add_X1 = X1;
        }
        
        // Undo lifting stages
        for (int i = 0; i < X0.rows; i++) {
            for (int j = 0; j < X0.cols; j++) {
                X0.at<double>(i, j) = wavelet_img.at<double>(i, j)/wp.scale_factor;
            }
        }
        X0 = X0 - filter(wp.lift_filter.col(3), add_X1, add_X1.row(0)*wp.lift_filter.at<double>(0,3));
        
        Mat X0_rightShift = Mat::zeros(rightShiftIdx.size(), X0.cols, CV_64FC1); // X0(rightShift,:,:)
        for (int i = 0; i < X0_rightShift.rows; i++) {
            for (int j = 0; j < X0_rightShift.cols; j++) {
                X0_rightShift.at<double>(i, j) = X0.at<double>(rightShiftIdx[i], j);
            }
        }
        add_X1 = add_X1 - filter(wp.lift_filter.col(2), X0_rightShift, X0.row(0)*wp.lift_filter.at<double>(0,2));
        X0 = X0 - filter(wp.lift_filter.col(1), add_X1, add_X1.row(0)*wp.lift_filter.at<double>(0,1));
        
        for (int i = 0; i < X0_rightShift.rows; i++) {
            for (int j = 0; j < X0_rightShift.cols; j++) {
                X0_rightShift.at<double>(i, j) = X0.at<double>(rightShiftIdx[i], j);
            }
        }
        add_X1 = add_X1 - filter(wp.lift_filter.col(0), X0_rightShift, X0.row(0)*wp.lift_filter.at<double>(0,0));
        
        Mat minus_X1;
        if (M1 % 2 == 1) {
            minus_X1 = Mat::zeros(add_X1.rows-1, add_X1.cols, CV_64FC1);
            for (int i = 0; i < minus_X1.rows; i++) {
                for (int j = 0; j < minus_X1.cols; j++) {
                    minus_X1.at<double>(i, j) = add_X1.at<double>(i, j);
                }
            }
        } else {
            minus_X1 = add_X1;
        }
        
        for (int i = 0; i < X0.rows; i++) {
            for (int j = 0; j < X0.cols; j++) {
                wavelet_img.at<double>(i*2, j) = X0.at<double>(i, j);
            }
        }
        for (int i = 0; i < minus_X1.rows; i++) {
            for (int j = 0; j < minus_X1.cols; j++) {
                wavelet_img.at<double>(i*2+1, j) = minus_X1.at<double>(i, j);
            }
        }
    }
}

void WaveletCdf97::TransferAlongRow(const int N1, const int N2, const int M1, const int M2, const bool flag) {
    
    // for forward transform
    if (flag == 0) {
        Mat X0 = Mat::zeros((N2-1)/2+1, N1, CV_64FC1);
        Mat X1 = Mat::zeros((N2-1)/2+1, N1, CV_64FC1);
        vector<int> rightShiftIdx;
        for (int i = 1; i < M2; ++i) {
            rightShiftIdx.push_back(i);
        } rightShiftIdx.push_back(M2-1);
        
        for (int i = 0; i < X0.rows; i++) {
            for (int j = 0; j < X0.cols; j++) {
                X0.at<double>(i, j) = wavelet_img.at<double>(j, i*2);
            }
        }
        
        // apply lifting stages
        Mat X0_rightShift = Mat::zeros(rightShiftIdx.size(), X0.cols, CV_64FC1); // X0(rightShift,:,:)
        for (int i = 0; i < X0_rightShift.rows; i++) {
            for (int j = 0; j < X0_rightShift.cols; j++) {
                X0_rightShift.at<double>(i, j) = X0.at<double>(rightShiftIdx[i], j);
            }
        }
     
        if ( N2 % 2 == 1) {
            Mat tmp = wavelet_img(cv::Rect(N2-3, 0, 1, N1))*wp.extrapolate_odd[0]
            + wavelet_img(cv::Rect(N2-2, 0, 1, N1))*wp.extrapolate_odd[1]
            + wavelet_img(cv::Rect(N2-1, 0, 1, N1))*wp.extrapolate_odd[2];
            
            for (int j = 0; j < X1.cols; j++) {
                for (int i = 0; i < X1.rows-1; i++) {
                    X1.at<double>(i, j) = wavelet_img.at<double>(j, i*2+1);
                }
                X1.at<double>(X1.rows-1, j) = tmp.at<double>(j, 0);
            }
            X1 = X1 + filter(wp.lift_filter.col(0), X0_rightShift, X0.row(0)*wp.lift_filter.at<double>(0,0));
            
        } else {
            for (int i = 0; i < X1.rows; i++) {
                for (int j = 0; j < X1.cols; j++) {
                    X1.at<double>(i, j) = wavelet_img.at<double>(j, i*2+1);
                }
            }
            X1 = X1 + filter(wp.lift_filter.col(0), X0_rightShift, X0.row(0)*wp.lift_filter.at<double>(0,0));
        }
        
        X0 = X0 + filter(wp.lift_filter.col(1), X1, X1.row(0)*wp.lift_filter.at<double>(0,1));
        
        for (int i = 0; i < X0_rightShift.rows; i++) {
            for (int j = 0; j < X0_rightShift.cols; j++) {
                X0_rightShift.at<double>(i, j) = X0.at<double>(rightShiftIdx[i], j);
            }
        }
        X1 = X1 + filter(wp.lift_filter.col(2), X0_rightShift, X0.row(0)*wp.lift_filter.at<double>(0,2));
        
        X0 = X0 + filter(wp.lift_filter.col(3), X1, X1.row(0)*wp.lift_filter.at<double>(0,3));
        
        Mat new_X1;
        if (N2 % 2 == 1) {
            new_X1 = Mat::zeros(X1.rows-1, X1.cols, CV_64FC1);
            for (int i = 0; i < new_X1.rows; i++) {
                for (int j = 0; j < new_X1.cols; j++) {
                    new_X1.at<double>(i, j) = X1.at<double>(i, j);
                }
            }
            
        } else {
            new_X1 = X1;
        }
        
        for (int i = 0; i < X0.rows; i++) {
            for (int j = 0; j < X0.cols; j++) {
                wavelet_img.at<double>(j, i) = X0.at<double>(i, j)*wp.scale_factor;
            }
        }
        
        
        for (int i = 0; i < new_X1.rows; i++) {
            for (int j = 0; j < new_X1.cols; j++) {
                wavelet_img.at<double>(j, i + X0.rows) = new_X1.at<double>(i, j)/wp.scale_factor;
            }
        }
        
        // for inverse transform
    } else {
        int Q = (int) ceil((double)(M2/2.0));
        Mat X0 = Mat::zeros(Q, M1, CV_64FC1);
        Mat X1 = Mat::zeros(M2-Q, M1, CV_64FC1);
        vector<int> rightShiftIdx;
        for (int i = 1; i < Q; ++i) {
            rightShiftIdx.push_back(i);
        } rightShiftIdx.push_back(Q-1);
        
        for (int i = 0; i < X1.rows; i++) {
            for (int j = 0; j < X1.cols; j++) {
                X1.at<double>(i, j) = wavelet_img.at<double>(j, i+Q)*wp.scale_factor;
            }
        }
        
        Mat add_X1;
        if (M2 % 2 == 1) {
            add_X1 = Mat::zeros(X1.rows+1, X1.cols, CV_64FC1);
            for (int i = 0; i < X1.rows; i++) {
                for (int j = 0; j < X1.cols; j++) {
                    add_X1.at<double>(i, j) = X1.at<double>(i, j);
                }
            }
            
        } else {
            add_X1 = X1;
        }
        
        // Undo lifting stages
        for (int i = 0; i < X0.rows; i++) {
            for (int j = 0; j < X0.cols; j++) {
                X0.at<double>(i, j) = wavelet_img.at<double>(j, i)/wp.scale_factor;
            }
        }
        X0 = X0 - filter(wp.lift_filter.col(3), add_X1, add_X1.row(0)*wp.lift_filter.at<double>(0,3));
        
        Mat X0_rightShift = Mat::zeros(rightShiftIdx.size(), X0.cols, CV_64FC1); // X0(rightShift,:,:)
        for (int i = 0; i < X0_rightShift.rows; i++) {
            for (int j = 0; j < X0_rightShift.cols; j++) {
                X0_rightShift.at<double>(i, j) = X0.at<double>(rightShiftIdx[i], j);
            }
        }
        add_X1 = add_X1 - filter(wp.lift_filter.col(2), X0_rightShift, X0.row(0)*wp.lift_filter.at<double>(0,2));
        X0 = X0 - filter(wp.lift_filter.col(1), add_X1, add_X1.row(0)*wp.lift_filter.at<double>(0,1));
        
        for (int i = 0; i < X0_rightShift.rows; i++) {
            for (int j = 0; j < X0_rightShift.cols; j++) {
                X0_rightShift.at<double>(i, j) = X0.at<double>(rightShiftIdx[i], j);
            }
        }
        add_X1 = add_X1 - filter(wp.lift_filter.col(0), X0_rightShift, X0.row(0)*wp.lift_filter.at<double>(0,0));
        
        Mat minus_X1;
        if (M2 % 2 == 1) {
            minus_X1 = Mat::zeros(add_X1.rows-1, add_X1.cols, CV_64FC1);
            for (int i = 0; i < minus_X1.rows; i++) {
                for (int j = 0; j < minus_X1.cols; j++) {
                    minus_X1.at<double>(i, j) = add_X1.at<double>(i, j);
                }
            }
        } else {
            minus_X1 = add_X1;
        }
        
        for (int i = 0; i < X0.rows; i++) {
            for (int j = 0; j < X0.cols; j++) {
                wavelet_img.at<double>(j, i*2) = X0.at<double>(i, j);
            }
        }
        
        for (int i = 0; i < minus_X1.rows; i++) {
            for (int j = 0; j < minus_X1.cols; j++) {
                wavelet_img.at<double>(j, i*2+1) = minus_X1.at<double>(i, j);
            }
        }
    }
}

Mat WaveletCdf97::filter(const Mat& B, const Mat& X, const Mat& Zi) {
    // b is a column vector
    Mat b;
    flip(B, b, 0);
    int padding = b.rows-1;
    Mat expendX = Mat::zeros(X.rows + padding, X.cols, CV_64FC1);
    Mat Y = Mat::zeros(X.rows, X.cols, CV_64FC1);
    
    // fill the expend X matrix
    for (int i = 0; i < X.rows; i++) {
        for (int j = 0; j < X.cols; j++) {
            expendX.at<double>(i+padding, j) = X.at<double>(i, j);
        }
    }
    
    // convolution operation
    for (int c = 0; c < expendX.cols; ++c) {
        for (int r = 0; r < expendX.rows-padding; ++r) {
            for (int i = 0; i < b.rows; i++) {
                Y.at<double>(r, c) += b.at<double>(i,0)*expendX.at<double>(r+i, c);
            }
        }
        // Zi is a row vector (initial delays)
        Y.at<double>(0, c) += Zi.at<double>(0, c);
    }
    
    return Y;
}