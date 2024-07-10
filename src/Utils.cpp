// src/Utils.cpp

#include "Utils.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <string>
#include <filesystem>

std::string image_results_path = "";
std::string image_results_filename = "";

void imageComparisonAndSave(cv::Mat& img1, cv::Mat& img2, std::string filename) {
    if (img1.size() != img2.size()) {
        std::cerr << "(overlayImages) Error: the sizes of the images do not match" << std::endl;
        return;
    }

    cv::Mat img1_converted, img2_converted;

    // Convert the grayscale image to a 3-channel image
    if (img1.type() == CV_8UC3 && img2.type() == CV_8UC1) {
        cv::cvtColor(img2, img2_converted, cv::COLOR_GRAY2BGR);
    } else {
        img2_converted = img2;
    }

    if (img1.type() == CV_8UC1 && img2.type() == CV_8UC3) {
        cv::cvtColor(img1, img1_converted, cv::COLOR_GRAY2BGR);
    } else {
        img1_converted = img1;
    }

    // Concatenate the images horizontally
    cv::Mat comparison;
    cv::hconcat(img1_converted, img2_converted, comparison);

    cv::imwrite(image_results_path + filename + image_results_filename + ".jpg", comparison);
}

void overlayImages(cv::Mat& img1, cv::Mat& img2, double alpha, double beta, cv::Mat& dst) {
    if (img1.size() != img2.size()) {
        std::cerr << "(overlayImages) Error: the sizes of the images do not match" << std::endl;
        return;
    }

    cv::Mat img1_converted, img2_converted;

    // Convert the grayscale image to a 3-channel image
    if (img1.type() == CV_8UC3 && img2.type() == CV_8UC1) {
        cv::cvtColor(img2, img2_converted, cv::COLOR_GRAY2BGR);
    } else {
        img2_converted = img2;
    }

    if (img1.type() == CV_8UC1 && img2.type() == CV_8UC3) {
        cv::cvtColor(img1, img1_converted, cv::COLOR_GRAY2BGR);
    } else {
        img1_converted = img1;
    }

    // Overlap images
    cv::addWeighted(img1_converted, alpha, img2_converted, beta, 0, dst);
}

void threshold(cv::Mat& src, cv::Mat& dst) {
    // Convert the image to grayscale
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Apply GaussianBlur to reduce noise and improve edge detection
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0, 0);

    // Apply Canny edge detection
    cv::Mat edges;
    cv::Canny(blurred, edges, 150, 200, 3, 0);

    // Convert the original image to HSV color space
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    // Define HSV range for lane colors (you may need to adjust these values)
    cv::Scalar lowerWhite = cv::Scalar(0, 0, 170);
    cv::Scalar upperWhite = cv::Scalar(255, 50, 255);

    // Define HSV range for yellow lanes (you may need to adjust these values)
    cv::Scalar lowerYellow = cv::Scalar(15, 100, 100);
    cv::Scalar upperYellow = cv::Scalar(45, 255, 255);

    // Threshold the HSV image to get only white colors
    cv::Mat whiteMask;
    cv::inRange(hsv, lowerWhite, upperWhite, whiteMask);
    
    // Threshold the HSV image to get only yellow colors
    cv::Mat yellowMask;
    cv::inRange(hsv, lowerYellow, upperYellow, yellowMask);
    
    // Combine the white and yellow masks
    cv::Mat combinedMask;
    cv::bitwise_or(whiteMask, yellowMask, combinedMask);

    // Combine Canny edges and HSV mask
    cv::Mat combined;
    cv::bitwise_or(edges, combinedMask, combined);

    // Copy the result to output image
    dst = combined.clone();

    if (SAVE_FLAG) {
        imageComparisonAndSave(src, edges, "orig_canny_comparison");
        imageComparisonAndSave(src, hsv, "orig_hsv_comparison");
        cv::imwrite(image_results_path + "hsv_white_mask" + image_results_filename + ".jpg", whiteMask);
        cv::imwrite(image_results_path + "hsv_yellow_mask" + image_results_filename + ".jpg", yellowMask);
        cv::imwrite(image_results_path + "hsv_combined_mask" + image_results_filename + ".jpg", combinedMask);
        imageComparisonAndSave(whiteMask, yellowMask, "white_yellow_hsv_masks_comparison");
        imageComparisonAndSave(src, combinedMask, "orig_hsv_mask_comparison");
        cv::imwrite(image_results_path + "final_threshold" + image_results_filename + ".jpg", dst);
    }
}

void calculateWarpPoints(const cv::Mat& image, std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dst) {
    // Calculate the vertices of the region of interest
    src.push_back(cv::Point2f(200, 720));
    src.push_back(cv::Point2f(1100, 720));
    src.push_back(cv::Point2f(550, 300));
    src.push_back(cv::Point2f(710, 300));

    // Calculate the destination points of the warp
    dst.push_back(cv::Point2f(300, 720));
    dst.push_back(cv::Point2f(980, 720));
    dst.push_back(cv::Point2f(300, 0));
    dst.push_back(cv::Point2f(980, 0));

    return;
}

void perspectiveTransform(std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dst, cv::Mat& H, cv::Mat& Hinv) {
    H = cv::getPerspectiveTransform(src, dst);
    Hinv = cv::getPerspectiveTransform(dst, src);
    
    return;
}

void perspectiveWarp(cv::Mat& image, cv::Mat& dst, cv::Mat& H) {
    cv::warpPerspective(image, dst, H, image.size(), cv::INTER_LINEAR);
    cv::imwrite(image_results_path + "warped" + image_results_filename + ".jpg", dst);

    return;
}

void calculateHistogram(cv::Mat& img, cv::Mat& histogram) {
    // Select bottom half of the image
    cv::Mat bottom_half = img.rowRange(img.rows / 2, img.rows);
    // Normalize pixel values, sum them up and reduce to a single row matrix
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

void plotHistogram(cv::Mat& histogram, cv::Mat& img, cv::Mat& dst) {
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

    if (PLOT_FLAG) {
        imshow("hist1", black_img);
        cv::waitKey(0);
    }
    
    dst = black_img;
}

// Function to perform polynomial fitting
cv::Mat polyfit(const std::vector<int>& x, const std::vector<int>& y, int degree) {
    // Number of data points
    int n = x.size();

    // Create the Vandermonde matrix
    cv::Mat A(n, degree + 1, CV_64F);
    cv::Mat Y(n, 1, CV_64F);

    for (int i = 0; i < n; ++i) {
        Y.at<double>(i, 0) = y[i];
        for (int j = 0; j <= degree; ++j) {
            A.at<double>(i, j) = pow(x[i], j);
        }
    }

    // Solve for the coefficients
    cv::Mat coefficients;
    cv::solve(A, Y, coefficients, cv::DECOMP_SVD);

    return coefficients;
}

void fitPolyToLaneLines(cv::Mat& warped, int left_peak, int right_peak, cv::Mat& left_fit, cv::Mat& right_fit) {
    // Number of sliding windows for lane pixels searching
    int num_of_windows = 10;
    // Height of each window
    int window_height = warped.rows / num_of_windows;

    // Non-zero points in the warped image
    std::vector<cv::Point> nonzero_points;
    // x and y coordinates of the non-zero points
    std::vector<double> nonzero_x, nonzero_y;

    // Find all non-zero points in the warped image
    cv::findNonZero(warped, nonzero_points);
    
    // Populate nonzero_x and nonzero_y vectors
    for (const auto& point : nonzero_points) {
        nonzero_x.push_back(point.x);
        nonzero_y.push_back(point.y);
    }

    // Width of the windows +- margin
    int margin {100};
    // Minimum pixel number to recenter windows
    int minpix {50};

    // Vectors for storing indices of points within windows
    std::vector<int> left_line_indices, right_line_indices;

    cv::Mat win_result_img(warped.size(), CV_8UC3, cv::Scalar(0,0,0));
 
    // Starting x positions of the windows
    int leftx_current = left_peak;
    int rightx_current = right_peak;

    for (int window = 0; window < num_of_windows; window++) { 
        // Set window boundaries
        int win_y_low = warped.rows - (window + 1) * window_height;
        int win_y_high = warped.rows - window * window_height;
        
        int win_xleft_low = leftx_current - margin;
        int win_xleft_high = leftx_current + margin;
        int win_xright_low = rightx_current - margin;
        int win_xright_high = rightx_current + margin;

        // Draw windows
        cv::rectangle(win_result_img, cv::Point(win_xleft_low, win_y_low), cv::Point(win_xleft_high, win_y_high), cv::Scalar(0,255,0), 2);
        cv::rectangle(win_result_img, cv::Point(win_xright_low, win_y_low), cv::Point(win_xright_high, win_y_high), cv::Scalar(0,255,0), 2);

        // Get points inside windows and store them
        std::vector<int> good_left_inds, good_right_inds;
        for (size_t i = 0; i < nonzero_x.size(); i++) {
            if (nonzero_y[i] >= win_y_low && nonzero_y[i] < win_y_high &&
                nonzero_x[i] >= win_xleft_low && nonzero_x[i] < win_xleft_high) {
                good_left_inds.push_back(i);
            }
            if (nonzero_y[i] >= win_y_low && nonzero_y[i] < win_y_high &&
                nonzero_x[i] >= win_xright_low && nonzero_x[i] < win_xright_high) {
                good_right_inds.push_back(i);
            }
        }

        // Append indices to the lists
        left_line_indices.insert(left_line_indices.end(), good_left_inds.begin(), good_left_inds.end());
        right_line_indices.insert(right_line_indices.end(), good_right_inds.begin(), good_right_inds.end());

        // If more good pixels then minpix are found, recenter next window based on the mean of these points
        if (good_left_inds.size() > minpix) {
            int sum = 0;
            for (int idx : good_left_inds) {
                sum += nonzero_x[idx];
            }
            leftx_current = sum / good_left_inds.size();
        }
        if (good_right_inds.size() > minpix) {
            int sum = 0;
            for (int idx : good_right_inds) {
                sum += nonzero_x[idx];
            }
            rightx_current = sum / good_right_inds.size();
        }
    }

    // Extract left and right line pixel positions
    std::vector<int> leftx, lefty, rightx, righty;
    for (int idx : left_line_indices) {
        leftx.push_back(nonzero_x[idx]);
        lefty.push_back(nonzero_y[idx]);
    }
    for (int idx : right_line_indices) {
        rightx.push_back(nonzero_x[idx]);
        righty.push_back(nonzero_y[idx]);
    }

    // Fit a second order polynomial to each
    left_fit = polyfit(lefty, leftx, 2);
    right_fit = polyfit(righty, rightx, 2);

    cv::Mat warped_with_windows;
    overlayImages(warped, win_result_img, 1, 1, warped_with_windows);

    if (PLOT_FLAG) {
        imshow("Warped image with windows", warped_with_windows);
        cv::waitKey(0);
    }

    if (SAVE_FLAG) {
        cv::imwrite(image_results_path + "warped_with_windows" + image_results_filename + ".jpg", warped_with_windows);
    }
}

void linesOnWarped(cv::Mat warped, std::vector<int> ploty, std::vector<int> left_fitx, std::vector<int> right_fitx) {
    // Generate black image and colour lane lines
    cv::Mat lane_lines(warped.size(), CV_8UC3, cv::Scalar(0, 0, 0));

    // Draw polyline on image
    std::vector<cv::Point> left_polyline, right_polyline;
    for (size_t i = 0; i < ploty.size(); ++i) {
        left_polyline.emplace_back(left_fitx[i], ploty[i]);
        right_polyline.emplace_back(right_fitx[i], ploty[i]);
    }

    const cv::Point *left_polyline_data = &left_polyline[0];
    const cv::Point *right_polyline_data = &right_polyline[0];

    int num_points = static_cast<int>(ploty.size());
    
    polylines(lane_lines, &left_polyline_data, &num_points, 1, false, cv::Scalar(0, 0, 255), 5);
    polylines(lane_lines, &right_polyline_data, &num_points, 1, false, cv::Scalar(0, 0, 255), 5);

    if (PLOT_FLAG) {
        cv::Mat warped_with_lines;
        overlayImages(lane_lines, warped, 1, 0.3, warped_with_lines);
        imshow("Warped image with lane lines", warped_with_lines);
        cv::waitKey(0);
    }
    
    if (SAVE_FLAG) {
        cv::Mat warped_with_lines;
        overlayImages(lane_lines, warped, 1, 0.3, warped_with_lines);
        cv::imwrite(image_results_path + "warped_with_lines" + image_results_filename + ".jpg", warped_with_lines);
        imageComparisonAndSave(warped, warped_with_lines, "warped_and_warped_lines_comparison");
    }
}

void markLane(cv::Mat initial_image, cv::Mat warped, cv::Mat Hinv, std::vector<int> ploty, std::vector<int> left_fitx, std::vector<int> right_fitx, cv::Mat& dst) {
    // Combine ploty and left_fitx, right_fitx into points for polylines
    std::vector<cv::Point> left_points, right_points;
    for (size_t i = 0; i < ploty.size(); ++i) {
        left_points.push_back(cv::Point(left_fitx[i], ploty[i]));
        right_points.push_back(cv::Point(right_fitx[i], ploty[i]));
    }

    // Create a polygon that covers the area between the lane lines
    std::vector<cv::Point> lane_polygon;
    lane_polygon.insert(lane_polygon.end(), left_points.begin(), left_points.end());
    lane_polygon.insert(lane_polygon.end(), right_points.rbegin(), right_points.rend());

    // Create an image to draw the filled polygon
    cv::Mat lane_area = cv::Mat::zeros(warped.size(), CV_8UC3);

    // Fill the polygon with a color (e.g., green with some transparency)
    cv::fillPoly(lane_area, std::vector<std::vector<cv::Point>>{lane_polygon}, cv::Scalar(170, 100, 230));

    // Warp the lane area back to the original image perspective
    cv::Mat new_warp;
    cv::warpPerspective(lane_area, new_warp, Hinv, initial_image.size());

    // Blend the lane area with the original image
    cv::addWeighted(initial_image, 1, new_warp, 0.3, 0, dst);

    if (PLOT_FLAG) {
        // Display the final image with the filled lane area
        cv::imshow("Lane Area on Original Image", dst);
        cv::waitKey(0);
    }

    if (SAVE_FLAG) {
        cv::imwrite(image_results_path + "marked_lane" + image_results_filename + ".jpg", dst);   
    }
}

void pipeline(cv::Mat& src, cv::Mat& dst) {
    // std::cout << image_results_path << std::endl;

    cv::Mat thresholded, abs_sobel;
    cv::Mat warped, unwarped; 
    std::vector<cv::Point2f> ROI_points, warp_destination_points;
    cv::Mat H, Hinv;          
    cv::Mat histogram_plot;

    threshold(src, thresholded);
    calculateWarpPoints(src, ROI_points, warp_destination_points);
    perspectiveTransform(ROI_points, warp_destination_points, H, Hinv); 
    perspectiveWarp(thresholded, warped, H);

    cv::Mat histogram;
    calculateHistogram(warped, histogram);

    plotHistogram(histogram, warped, histogram_plot);

    int left_peak, right_peak;
    findLaneHistogramPeaks(histogram, left_peak, right_peak);
    
    cv::Mat left_fit, right_fit;
    fitPolyToLaneLines(warped, left_peak, right_peak, left_fit, right_fit);

    // Generate x and y values for plotting
    std::vector<int> ploty(warped.rows);
    for (size_t i = 0; i < ploty.size(); ++i) {
        ploty[i] = static_cast<int>(i);
    }
    std::vector<int> left_fitx(ploty.size());
    std::vector<int> right_fitx(ploty.size());
    for (size_t i = 0; i < ploty.size(); ++i) {
        left_fitx[i] = left_fit.at<double>(2, 0) * ploty[i] * ploty[i] +
                    left_fit.at<double>(1, 0) * ploty[i] +
                    left_fit.at<double>(0, 0);
        right_fitx[i] = right_fit.at<double>(2, 0) * ploty[i] * ploty[i] +
                        right_fit.at<double>(1, 0) * ploty[i] +
                        right_fit.at<double>(0, 0);
    }

    linesOnWarped(warped, ploty, left_fitx, right_fitx);
    markLane(src, warped, Hinv, ploty, left_fitx, right_fitx, dst);

    if (PLOT_FLAG) {
        imshow("Thresholded", thresholded);
        cv::waitKey(0);
        imshow("Warped", warped);
        cv::waitKey(0);    
    }

    if (SAVE_FLAG) {
        cv::imwrite(image_results_path + "histogram" + image_results_filename + ".jpg", histogram_plot); 
        cv::imwrite(image_results_path + "marked_lane" + image_results_filename + ".jpg", dst);  
        imageComparisonAndSave(src, thresholded, "threshold_comparison");
        imageComparisonAndSave(thresholded, warped, "warped_comparison");
        imageComparisonAndSave(histogram_plot, warped, "histogram_warped_comparison");
    }
}

std::string extractFilename(const std::string &path) {
    // Find the last position of '/'
    int last_slash_idx = path.find_last_of("/\\");
    std::string filename;

    if (last_slash_idx != std::string::npos) {
        filename = path.substr(last_slash_idx + 1);
    } 
    else {
        filename = path;
    }

    int period_idx = filename.rfind('.');
    if (period_idx != std::string::npos) {
        return filename.substr(0, period_idx);
    }

    return filename;
}