#ifndef __UTILS_H__
#define __UTILS_H__

#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>

void calculateHistogram(cv::Mat& img, cv::Mat& histogram)
{
    cv::Mat bottom_half = img.rowRange(img.rows / 2, img.cols / 2);
    cv::reduce(bottom_half / 255, histogram, 0, cv::REDUCE_SUM, CV_32S);
}

std::vector<double> linspace(double a, double b, int n) {
    std::vector<double> array;
    double step = (b - a) / (n - 1);

    while (a <= b) {
        array.push_back(a);
        a += step;
    }

    return array;
} 

void findLaneHistogramPeaks(cv::Mat& histogram, int& left_half_peak, int& right_half_peak) {
    int x_middle_point = histogram.cols / 2;

    left_half_peak = static_cast<int>(std::distance(histogram.begin<int>(), std::max_element(histogram.begin<int>(), histogram.begin<int>() + x_middle_point)));
    right_half_peak = x_middle_point + static_cast<int>(std::distance(histogram.begin<int>() + x_middle_point, std::max_element(histogram.begin<int>() + x_middle_point, histogram.end<int>())));

    return;
}

void lineFit() {


    return;
}

void plotHistogram(cv::Mat& histogram, cv::Mat& img) {
    // Generating x and y axis values for plots
    std::vector<double> fitx, fity, hist_fity;
    fity = linspace(0, img.rows - 1, img.rows);
    fitx = linspace(0, img.cols - 1, img.cols);

    // 
    for (int i = 0; i < histogram.cols; i++) {
        hist_fity.push_back(img.rows - histogram.at<int>(0, i));
    }

    // Blank image for histogram plot
    cv::Mat black_img(img.size(), CV_8UC3, cv::Scalar(0,0,0));

    std::vector<cv::Point2f> points;
    for (int i = 0; i < hist_fity.size(); i++) {
        points.push_back(cv::Point2f(static_cast<float>(fitx[i]), static_cast<float>(hist_fity[i])));
    }

    cv::Mat curve(points, true);
    curve.convertTo(curve, CV_32S);
    cv::Scalar red(0, 0, 255);
    cv::polylines(black_img, curve, false, red, 2);

    imshow("hist1", black_img);
    cv::waitKey(0);
}

#endif