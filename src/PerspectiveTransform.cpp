#include "PerspectiveTransform.hpp"

void PerspectiveTransform::calculateWarpPoints(const cv::Mat& image, std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dst, int y_bottom, int y_top, int offset)
{
    // Calculate the vertices of the region of interest
    src.push_back(cv::Point2f(200, 720));
    src.push_back(cv::Point2f(1100, 720));
    src.push_back(cv::Point2f(595, 450));
    src.push_back(cv::Point2f(685, 450));

    // Calculate the destination points of the warp
    dst.push_back(cv::Point2f(300, 720));
    dst.push_back(cv::Point2f(980, 720));
    dst.push_back(cv::Point2f(300, 0));
    dst.push_back(cv::Point2f(980, 0));

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