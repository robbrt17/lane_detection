#ifndef __UTILS_H__
#define __UTILS_H__

#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace cv;

void detectEdges(cv::Mat& src, cv::Mat& dst) {
    cv::Mat hls;
    cv::cvtColor(src, hls, cv::COLOR_RGB2HLS);

    std::vector<cv::Mat> hls_channels;
    cv::split(hls, hls_channels);
    cv::Mat s_channel = hls_channels[2];

    cv::Mat hls_binary_output = cv::Mat::zeros(s_channel.size(), CV_8U);
    cv::inRange(s_channel, cv::Scalar(171), cv::Scalar(255), hls_binary_output);

    cv::Mat gray;
    cv::cvtColor(hls, gray, cv::COLOR_BGR2GRAY);

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    cv::Mat canny;
    cv::Canny(blurred, canny, 100, 200);

    dst = canny | hls_binary_output;
}

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

// cv::Mat polyfit(const std::vector<int>& x, const std::vector<int>& y, int degree) {
//     // Number of data points
//     int n = x.size();

//     // Create the Vandermonde matrix
//     cv::Mat A(n, degree + 1, CV_64F);
//     cv::Mat Y(n, 1, CV_64F);

//     // Fill the Vandermonde matrix and the Y matrix
//     for (int i = 0; i < n; ++i) {
//         Y.at<double>(i, 0) = y[i];
//         for (int j = 0; j <= degree; ++j) {
//             A.at<double>(i, j) = pow(x[i], j);
//         }
//     }

//     // Solve for the coefficients using the normal equations
//     cv::Mat A_transpose, A_transpose_A, A_transpose_Y;
//     transpose(A, A_transpose);
//     A_transpose_A = A_transpose * A;
//     A_transpose_Y = A_transpose * Y;

//     cv::Mat coefficients;
//     solve(A_transpose_A, A_transpose_Y, coefficients, cv::DECOMP_SVD);

//     return coefficients;
// }

// Function to perform polynomial fitting
Mat polyfit(const std::vector<int>& x, const std::vector<int>& y, int degree) {
    // Number of data points
    int n = x.size();

    // Create the Vandermonde matrix
    Mat A(n, degree + 1, CV_64F);
    Mat Y(n, 1, CV_64F);

    for (int i = 0; i < n; ++i) {
        Y.at<double>(i, 0) = y[i];
        for (int j = 0; j <= degree; ++j) {
            A.at<double>(i, j) = pow(x[i], j);
        }
    }

    // Solve for the coefficients
    Mat coefficients;
    solve(A, Y, coefficients, DECOMP_SVD);

    return coefficients;
}

#endif