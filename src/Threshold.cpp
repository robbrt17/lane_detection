#include "Threshold.hpp"

void Threshold::absoluteSobel(cv::Mat const& src, cv::Mat& dest, char orient, int kernel_size, int thresh_min, int thresh_max)
{
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_RGB2GRAY);

    cv::Mat sobel, abs_sobel, scaled_sobel;

    // cv::Sobel(gray, sobel_x, CV_64F, 1, 0, 9);
    // cv::Sobel(gray, sobel_y, CV_64F, 0, 1, 9);

    // cv::Mat sobel_mag = cv::Mat(sobel_x.rows, sobel_x.cols, CV_64F, 0.0);

    // for (int i = 0; i < sobel_mag.rows; i++) {
    //     for (int j = 0; j < sobel_mag.cols; j++) {
    //         sobel_mag.at<double>(i,j) = sqrt(pow(sobel_x.at<double>(i,j), 2) + pow(sobel_y.at<double>(i,j), 2));
    //     }
    // }   

    // double min, max;
    // cv::minMaxLoc(sobel_mag, &min, &max);
    // sobel_mag = (sobel_mag / max) * 255;
    // sobel_mag.convertTo(sobel_mag, CV_8UC1);

    // cv::threshold(sobel_mag, dest, 50, 1, cv::THRESH_BINARY);

    // Compute the gradient along x or y axis
    if (orient == 'x') {
		cv::Sobel(gray, sobel, CV_64F, 1, 0);
	}
	else {
        cv::Sobel(gray, sobel, CV_64F, 0, 1);
	}

    // Get absolute value
    // abs_sobel = cv::abs(sobel);
    cv::convertScaleAbs(sobel, abs_sobel);

    // Since the gradients are of type "float", convert back to uint8 so it can be usable in other OpenCV functions
	// double min, max;
	// cv::minMaxLoc(abs_sobel, &min, &max);
	// scaled_sobel = 255 * (abs_sobel / max);
	// scaled_sobel.convertTo(scaled_sobel, CV_8UC1);

    cv::inRange(abs_sobel, thresh_min, thresh_max, dest);
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

void Threshold::hsv(cv::Mat const& src, cv::Mat& dst, uint8_t threshold_min, uint8_t threshold_max)
{
    cv::Mat hsv_img;
    cv::cvtColor(src, hsv_img, cv::COLOR_BGR2HSV);

    cv::inRange(hsv_img, threshold_min, threshold_max, dst);
}

void Threshold::binaryEqualizedGrayscale(cv::Mat const& src, cv::Mat& dst)
{
    cv::Mat gray, eq;
    cv::cvtColor(src, gray, cv::COLOR_RGB2GRAY);

    cv::equalizeHist(gray, eq);

    cv::threshold(eq, dst, 250, 255, cv::THRESH_BINARY);
}

void Threshold::combined(cv::Mat const& img, cv::Mat& dst)
{
    cv::Mat hls_bin, sobel, white_mask;

    Threshold::hls(img, hls_bin, 170, 255);
    // Threshold::absoluteSobel(img, sobel, 'x', 3, 50, 255);
    Threshold::canny(img, sobel);
    // Threshold::binaryEqualizedGrayscale(img, white_mask);

    dst = hls_bin | sobel;

    // cv::Mat sobel_x, sobel_y, combined, hls_bin;
    // // Threshold::absoluteSobel(img, sobel_x, 'x', 3, 50, 255);
    // Threshold::canny(img, sobel_x);

    // Threshold::hls(img, hls_bin, 170, 255);

    // dst = sobel_x | hls_bin;
    
    // return;
}