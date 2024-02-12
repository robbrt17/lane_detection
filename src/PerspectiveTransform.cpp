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

void perspectiveToMaps(const cv::Mat& perspective_mat, const cv::Size& img_size, cv::Mat& map1, cv::Mat& map2)
{
    cv::Mat inv_perspective(perspective_mat.inv());
    inv_perspective.convertTo(inv_perspective, CV_32FC1);

    cv::Mat xy(img_size, CV_32FC2);
    float *pxy = (float*)xy.data;
    for (int y = 0; y < img_size.height; y++)
        for (int x = 0; x < img_size.width; x++)
        {
            *pxy++ = x;
            *pxy++ = y;
        }

    // perspective transformation of the points
    cv::Mat xy_transformed;
    cv::perspectiveTransform(xy, xy_transformed, inv_perspective);

    // split x/y to extra maps
    assert(xy_transformed.channels() == 2);
    cv::Mat maps[2]; // map_x, map_y
    cv::split(xy_transformed, maps);

    // remap() with integer maps is faster
    cv::convertMaps(maps[0], maps[1], map1, map2, CV_16SC2);
}