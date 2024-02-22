#ifndef __THRESHOLD_H__
#define __THRESHOLD_H__

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/calib3d/calib3d_c.h>
#include <cmath>

class Threshold
{
    public:
        static void absoluteSobel(cv::Mat const& src, cv::Mat& dest, char orient = 'x', int kernel_size = 3, int thresh_min = 0, int thresh_max = 255);
        static void hls(cv::Mat const& src, cv::Mat& dst, uint8_t threshold_min = 100, uint8_t threshold_max = 255);
        static void hsv(cv::Mat const& src, cv::Mat& dst, uint8_t threshold_min = 100, uint8_t threshold_max = 255);
        static void binaryEqualizedGrayscale(cv::Mat const& src, cv::Mat& dst);
        static void canny(cv::Mat const& src, cv::Mat& dst);
        void combined(cv::Mat const& img, cv::Mat& dst);
};

#endif