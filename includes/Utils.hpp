// includes/Utils.hpp
#ifndef __UTILS_H__
#define __UTILS_H__

#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <string>

#define PLOT_FLAG 0
#define SAVE_FLAG 0

extern std::string image_results_path;
extern std::string image_results_filename;

void imageComparisonAndSave(cv::Mat& img1, cv::Mat& img2, std::string filename);

void overlayImages(cv::Mat& img1, cv::Mat& img2, double alpha, double beta, cv::Mat& dst);

void threshold(cv::Mat& src, cv::Mat& dst);

void calculateWarpPoints(const cv::Mat& image, std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dst);

void perspectiveTransform(std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dst, cv::Mat& H, cv::Mat& Hinv);

void perspectiveWarp(cv::Mat& image, cv::Mat& dst, cv::Mat& H);

void calculateHistogram(cv::Mat& img, cv::Mat& histogram);

std::vector<double> linspace(double a, double b, int n);

void findLaneHistogramPeaks(cv::Mat& histogram, int& left_half_peak, int& right_half_peak);

void plotHistogram(cv::Mat& histogram, cv::Mat& img, cv::Mat& dst);

cv::Mat polyfit(const std::vector<int>& x, const std::vector<int>& y, int degree);

void fitPolyToLaneLines(cv::Mat& warped, int left_peak, int right_peak, cv::Mat& left_fit, cv::Mat& right_fit);

void linesOnWarped(cv::Mat warped, std::vector<int> ploty, std::vector<int> left_fitx, std::vector<int> right_fitx);

void markLane(cv::Mat initial_image, cv::Mat warped, cv::Mat Hinv, std::vector<int> ploty, std::vector<int> left_fitx, std::vector<int> right_fitx, cv::Mat& dst);

void pipeline(cv::Mat& src, cv::Mat& dst);

std::string extractFilename(const std::string &path);

#endif