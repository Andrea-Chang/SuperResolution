//
//  interpolator.cpp
//  SuperResolution
//
//  Created by Hana_Chang on 2015/5/14.
//  Copyright (c) 2015年 Andrea.C. All rights reserved.
//

#include "interpolator.h"
using cv::Mat;
using cv::Point2d;




Interpolator::Interpolator(const Mat& src, double scale) {
    Mat src64;
    src.convertTo(src64, CV_64FC1);
    src64.copyTo(this->src);
    this->scale = scale;
}

Mat Interpolator::BicubicInterpolate() {
    Mat dst = Mat::zeros(src.rows*scale, src.cols*scale, CV_64FC1);
    Point2d p;
    
    for (int r = 0; r < dst.rows; r++) {
        for (int c = 0; c < dst.cols; c++) {
            p.x = c;
            p.y = r;
            dst.at<double>(r, c) = GetColor(p);
        }
    }
    return dst;
}

double Interpolator::GetColor(const Point2d& p) {
    double a = -0.5; // bicubic coefficient
    Point2d pivot;
    int x_start = 0;
    int y_start = 0;
    int   x_end = 0;
    int   y_end = 0;
    
    pivot.x = (src.cols-1)/(src.cols*scale-1)*p.x;
    pivot.y = (src.rows-1)/(src.rows*scale-1)*p.y;
    
    // if pivot.x is not an integer
    if ( (floor(pivot.x) - ceil(pivot.x)) > DBL_MIN ) {
        x_start = (int) floor(pivot.x) - 1;
        x_end   = (int) ceil(pivot.x) + 1;
    } else {
        x_start = (int) pivot.x - 1;
        x_end   = (int) pivot.x + 2;
    }
    
    // if pivot.y is not an integer
    if ( (floor(pivot.y) - ceil(pivot.y)) > DBL_MIN ) {
        y_start = (int) floor(pivot.y) - 1;
        y_end   = (int) ceil(pivot.y) + 1;
    } else {
        y_start = (int) pivot.y - 1;
        y_end   = (int) pivot.y + 2;
    }
    
    // get neighbors': corrdinate, weight, color
    Point5d neighbor;
    double x_diff, y_diff;
    for (int i = x_start; i <= x_end; i++) {
        for (int j = y_start; j <= y_end; j++) {
            neighbor.x = i;
            neighbor.y = j;
            x_diff = std::abs(pivot.x-neighbor.x);
            y_diff = std::abs(pivot.y-neighbor.y);
            
            // compute weight of x direction
            if ( x_diff <= 1.0 ) {
                neighbor.xw = (a+2)*pow(x_diff,3) - (a+3)*pow(x_diff,2) + 1;
            } else if ( 1.0 < x_diff && x_diff < 2.0 ) {
                neighbor.xw = a*pow(x_diff,3) - 5*a*pow(x_diff,2) + 8*a*x_diff - 4*a;
            } else {
                neighbor.xw = 0;
            }
            
            // compute weight of y direction
            if ( y_diff <= 1.0 ) {
                neighbor.yw = (a+2)*pow(y_diff,3) - (a+3)*pow(y_diff,2) + 1;
            } else if ( 1.0 < y_diff && y_diff < 2.0 ) {
                neighbor.yw = a*pow(y_diff,3) - 5*a*pow(y_diff,2) + 8*a*y_diff - 4*a;
            } else {
                neighbor.yw = 0;
            }
            
            double index_x = 0.0;
            double index_y = 0.0;
            // get color of each neighbor
            if (neighbor.x < 0) {
                index_x = -neighbor.x-1;
            } else if (neighbor.x >= 0 && neighbor.x < src.cols-1) {
                index_x = neighbor.x;
            } else {
                index_x = 2*src.cols-2-neighbor.x;
            }
            
            if (neighbor.y < 0) {
                index_y = -neighbor.y-1;
            } else if (neighbor.y >= 0 && neighbor.y < src.rows-1) {
                index_y = neighbor.y;
            } else {
                index_y = 2*src.rows-2-neighbor.y;
            }
            
            neighbor.color = src.at<double>(index_y, index_x);
            neighbors.push_back(neighbor);
        }
    }
    
    double color = 0.0;
    for (int i = 0; i < neighbors.size(); i++) {
        color += neighbors[i].xw*neighbors[i].yw*neighbors[i].color;
    }
    
    neighbors.clear();
    return color;
}