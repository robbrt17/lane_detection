#ifndef __CAMERACALIBRATION_H__
#define __CAMERACALIBRATION_H__

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/calib3d/calib3d_c.h>

class CameraCalibration 
{
    // 3d points in real world space
    std::vector<std::vector<cv::Point3f>> objectPoints;
    // 2d points in image plane
    std::vector<std::vector<cv::Point2f>> imagePoints;

    cv::Mat cameraMatrix, distortionCoefficients;

    int calibrationDoneFlag;

    public:
        void addChessboardPoints(cv::Size& chessboardSize, bool showPoints);
        double calibrate(cv::Size& imageSize);
        cv::Mat undistort(cv::Mat& image);
        void saveCalibration(std::string const& filename);
        void displayUndistortedImages();

        void start(cv::Size chessboardSize);
};

#endif