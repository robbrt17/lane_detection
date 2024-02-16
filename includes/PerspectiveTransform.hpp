#ifndef __PERSPECTIVETRANSFORM_H__
#define __PERSPECTIVETRANSFORM_H__

#include <iostream>
#include <opencv2/opencv.hpp>

class PerspectiveTransform 
{
    public:
        void calculateWarpPoints(const cv::Mat& image, std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dst, int y_bottom, int y_top, int offset = 200);
        void perspectiveTransform(std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dst, cv::Mat& M, cv::Mat& Minv);
        void perspectiveWarp(cv::Mat& image, cv::Mat& dst, cv::Mat& M);
};

#endif