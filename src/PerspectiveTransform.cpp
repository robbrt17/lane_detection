#include "PerspectiveTransform.hpp"

void PerspectiveTransform::calculateWarpPoints(const cv::Mat& image, std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dst)
{
    int h = image.rows;
    int w = image.cols;

    // Calculate the vertices of the region of interest
    src.push_back(cv::Point2f(w, h-10));
    src.push_back(cv::Point2f(0, h-10));
    // src.push_back(cv::Point2f(546, 460));
    // src.push_back(cv::Point2f(732, 460));
    src.push_back(cv::Point2f(354, 305));
    src.push_back(cv::Point2f(500, 305));

    // Calculate the destination points of the warp
    dst.push_back(cv::Point2f(w, h));
    dst.push_back(cv::Point2f(0, h));
    dst.push_back(cv::Point2f(0, 0));
    dst.push_back(cv::Point2f(w, 0));

    return;
}

void PerspectiveTransform::perspectiveTransform(std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dst, cv::Mat& M, cv::Mat& Minv)
{
    M = cv::getPerspectiveTransform(src, dst);
    Minv = cv::getPerspectiveTransform(dst, src);
    
    return;
}

void PerspectiveTransform::perspectiveWarp(cv::Mat& image, cv::Mat& dst, cv::Mat& M)
{
    cv::warpPerspective(image, dst, M, image.size(), cv::INTER_LINEAR);

    return;
}