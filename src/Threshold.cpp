#include "Threshold.hpp"

void Threshold::absoluteSobel(cv::Mat const& src, cv::Mat& dest, char orient, int kernel_size, int thresh_min, int thresh_max)
{
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_RGB2GRAY);

    cv::Mat sobel, abs_sobel, scaled_sobel;

    // Compute the gradient along x or y axis
    if (orient == 'x') {
		cv::Sobel(gray, sobel, CV_64F, 1, 0);
	}
	else {
        cv::Sobel(gray, sobel, CV_64F, 0, 1);
	}

    // Get absolute value
    abs_sobel = cv::abs(sobel);

    // Since the gradients are of type "float", convert back to uint8 so it can be usable in other OpenCV functions
	double min, max;
	cv::minMaxLoc(abs_sobel, &min, &max);
	scaled_sobel = 255 * (abs_sobel / max);
	scaled_sobel.convertTo(scaled_sobel, CV_8UC1);

    cv::inRange(scaled_sobel, thresh_min, thresh_max, dest);
}

void Threshold::canny(cv::Mat const& src, cv::Mat& dst)
{
    cv::Mat blured;
    cv::GaussianBlur(src, blured, cv::Size(3, 3), 0, 0);
    cv::Canny(blured, dst, 100, 200, 3, false);
}

void Threshold::hls(cv::Mat const& src, cv::Mat& dst, uint8_t threshold_min, uint8_t threshold_max)
{
    cv::Mat hls_img;
    cv::cvtColor(src, hls_img, cv::COLOR_BGR2HLS);

    cv::Mat hls_channels[3];
    cv::split(hls_img, hls_channels);

    cv::inRange(hls_channels[2], 170, 255, dst);
}

void Threshold::combined(cv::Mat const& img, cv::Mat& dst)
{
    cv::Mat sobel_x, sobel_y, combined, hls_bin;
    // Threshold::absoluteSobel(img, sobel_x, 'x', 3, 50, 255);
    Threshold::canny(img, sobel_x);

    Threshold::hls(img, hls_bin, 170, 255);

    dst = sobel_x | hls_bin;
    
    return;
}