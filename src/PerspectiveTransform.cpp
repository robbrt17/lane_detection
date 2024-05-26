#include "PerspectiveTransform.hpp"

void calculateWarpPoints(const cv::Mat& image, std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dst)
{
    int h = image.rows;
    int w = image.cols;

    // // Calculate the vertices of the region of interest
    // src.push_back(cv::Point2f(w, h-5));
    // src.push_back(cv::Point2f(0, h-5));
    // // src.push_back(cv::Point2f(546, 460));
    // // src.push_back(cv::Point2f(732, 460));
    // src.push_back(cv::Point2f(358, 305));
    // src.push_back(cv::Point2f(496, 305));

    // // Calculate the destination points of the warp
    // dst.push_back(cv::Point2f(w, h));
    // dst.push_back(cv::Point2f(0, h));
    // dst.push_back(cv::Point2f(0, 0));
    // dst.push_back(cv::Point2f(w, 0));

    // Calculate the vertices of the region of interest
    src.push_back(cv::Point2f(200, 720));
    src.push_back(cv::Point2f(1100, 720));
    // src.push_back(cv::Point2f(546, 460));
    // src.push_back(cv::Point2f(732, 460));
    src.push_back(cv::Point2f(595, 450));
    src.push_back(cv::Point2f(685, 450));

    // Calculate the destination points of the warp
    dst.push_back(cv::Point2f(300, 720));
    dst.push_back(cv::Point2f(980, 720));
    dst.push_back(cv::Point2f(300, 0));
    dst.push_back(cv::Point2f(980, 0));

    return;
}

void perspectiveTransform(std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dst, cv::Mat& M, cv::Mat& Minv)
{
    M = cv::getPerspectiveTransform(src, dst);
    Minv = cv::getPerspectiveTransform(dst, src);
    
    return;
}

void perspectiveWarp(cv::Mat& image, cv::Mat& dst, cv::Mat& M)
{
    cv::warpPerspective(image, dst, M, image.size(), cv::INTER_LINEAR);

    return;
}