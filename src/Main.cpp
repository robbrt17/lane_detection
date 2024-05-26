#include <iostream>
#include <stdlib.h>
#include <string>
#include <thread>
#include <mutex>
#include <queue>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include "CameraCalibration.hpp"
#include "PerspectiveTransform.hpp"
#include "Threshold.hpp"
#include "QueueFPS.hpp"
#include "Utils.hpp"

using namespace std;
using namespace cv;

// #define USE_MULTIPLE_THREADS
// #define USE_2_THREADS
// #define NO_THREADS
// #define TEST_ON_IMAGE

int thread_number = 3;

cv::Mat thresholded, abs_sobel;
cv::Mat warped, unwarped;   

int main() 
{
    // Get test video
    cv::VideoCapture video_capture("./test_videos/project_video.mp4");

    // If it can't be opened, return error message and exit
    if (video_capture.isOpened() == false) 
    {
        std::cout << "THE VIDEO FILE COULD NOT BE OPENED OR WRONG FORMAT!" << std::endl;
        std::cin.get();
        return -1;
    }

    // Display number of FPS
    std::cout << "Video frames per second: " << video_capture.get(cv::CAP_PROP_FPS) << std::endl;


#ifdef USE_2_THREADS

    bool process = true;

    cv::VideoWriter videoWriter;
    double fps = 30.0;
    cv::Size frame_size(1280, 720);
    videoWriter.open("output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frame_size);

    if (!videoWriter.isOpened()) {
        std::cerr << "Error: Could not open the output video file for write\n";
    }

    QueueFPS<cv::Mat> framesQueue;
    std::thread framesThread([&](){
        cv::Mat frame;
        while (process)
        {
            video_capture >> frame;
            if (!frame.empty())
            {
                framesQueue.push(frame.clone());
            }
            else
            {
                break;
            }
        }
    });

    QueueFPS<cv::Mat> processedFramesQueue;
    auto processFrame = [&]() {
        while (process)
        {
            cv::Mat frame;

            if (!framesQueue.empty())
            {
                frame = framesQueue.get();
            }

            if (!frame.empty())
            {
                std::vector<cv::Point2f> ROI_points, warp_destination_points;
                cv::Mat M, Minv;          
                // combined(img, thresholded);
                combined(frame, thresholded);
                calculateWarpPoints(frame, ROI_points, warp_destination_points);
                perspectiveTransform(ROI_points, warp_destination_points, M, Minv); 
                perspectiveWarp(thresholded, warped, M);

                // imshow("Thresholded", thresholded);
                // cv::waitKey(0);
                // imshow("Warped", warped);
                // cv::waitKey(0);

                cv::Mat histogram;
                calculateHistogram(warped, histogram);

                int left_peak, right_peak;
                findLaneHistogramPeaks(histogram, left_peak, right_peak);

                // plotHistogram(histogram, warped);

                std::vector<double> fity;
                fity = linspace(0, warped.rows - 1, warped.rows);

                int num_of_windows = 9;
                int window_height = warped.rows / num_of_windows;

                std::vector<cv::Point> nonzero_points;
                cv::findNonZero(warped, nonzero_points);
                std::vector<double> nonzero_x, nonzero_y;

                for (const auto& point : nonzero_points) {
                    nonzero_x.push_back(point.x);
                    nonzero_y.push_back(point.y);
                }

                // Width of the windows +- margin
                int margin {50};
                // Minimum pixel number to recenter windows
                int minpix {25};

                std::vector<int> left_line_indices, right_line_indices;

                cv::Mat win_result_img(warped.size(), CV_8UC3, cv::Scalar(0,0,0));
                // cv::cvtColor(warped, win_result_img, cv::COLOR_GRAY2BGR);

                cv::Mat good_left_inds, good_right_inds;
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
                vector<int> leftx, lefty, rightx, righty;
                for (int idx : left_line_indices) {
                    leftx.push_back(nonzero_x[idx]);
                    lefty.push_back(nonzero_y[idx]);
                }
                for (int idx : right_line_indices) {
                    rightx.push_back(nonzero_x[idx]);
                    righty.push_back(nonzero_y[idx]);
                }

                // Fit a second order polynomial to each
                Mat left_fit = polyfit(lefty, leftx, 2);
                Mat right_fit = polyfit(righty, rightx, 2);

                // Generate x and y values for plotting
                vector<int> ploty(warped.rows);
                for (size_t i = 0; i < ploty.size(); ++i) {
                    ploty[i] = static_cast<int>(i);
                }
                vector<int> left_fitx(ploty.size());
                vector<int> right_fitx(ploty.size());
                for (size_t i = 0; i < ploty.size(); ++i) {
                    left_fitx[i] = left_fit.at<double>(2, 0) * ploty[i] * ploty[i] +
                                left_fit.at<double>(1, 0) * ploty[i] +
                                left_fit.at<double>(0, 0);
                    right_fitx[i] = right_fit.at<double>(2, 0) * ploty[i] * ploty[i] +
                                    right_fit.at<double>(1, 0) * ploty[i] +
                                    right_fit.at<double>(0, 0);
                }

                // Generate black image and colour lane lines
                Mat lane_lines(warped.size(), CV_8UC3, Scalar(0, 0, 0));

                // Draw polyline on image
                vector<Point> left_polyline, right_polyline;
                for (size_t i = 0; i < ploty.size(); ++i) {
                    left_polyline.emplace_back(left_fitx[i], ploty[i]);
                    right_polyline.emplace_back(right_fitx[i], ploty[i]);
                }
                const Point *left_polyline_data = &left_polyline[0];
                const Point *right_polyline_data = &right_polyline[0];
                int num_points = static_cast<int>(ploty.size());
                polylines(lane_lines, &left_polyline_data, &num_points, 1, false, Scalar(0, 255, 0), 5);
                polylines(lane_lines, &right_polyline_data, &num_points, 1, false, Scalar(0, 255, 0), 5);

                // Display results
                // imshow("Lane Lines", lane_lines);
                // waitKey(0);

                // Convert ploty and left_fitx, right_fitx to cv::Mat for warping
                cv::Mat plotyMat = cv::Mat(ploty).reshape(1);
                cv::Mat left_fitxMat = cv::Mat(left_fitx).reshape(1);
                cv::Mat right_fitxMat = cv::Mat(right_fitx).reshape(1);

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
                cv::warpPerspective(lane_area, new_warp, Minv, frame.size());

                // Blend the lane area with the original image
                cv::Mat result;
                cv::addWeighted(frame, 1, new_warp, 0.3, 0, result);

                // Display the final image with the filled lane area
                // cv::imshow("Lane Area on Original Image", result);
                // cv::waitKey(0);

                processedFramesQueue.push(result);
            }
        }
    };

    // Thread pool for processing frames
    const int num_processing_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> processing_threads;

    for (int i = 0; i < num_processing_threads; ++i) {
        processing_threads.emplace_back(processedFramesQueue);
    }

    std::thread displayThread([&](){
        while (true) {
            cv::Mat frame;
            if (!processedFramesQueue.empty()) {
                frame = processedFramesQueue.get();
                std::cout << "Frames Queue FPS:" << framesQueue.getFPS() << std::endl;
                std::cout << "Processed Frames Queue FPS:" << processedFramesQueue.getFPS() << std::endl;
                std::cout << "==============================" << std::endl;

                // imshow("test", frame);

                videoWriter.write(frame);
            }

            if (cv::waitKey(1) == 'q') {
                break;
            }
        }

        process = false;
        framesThread.join();
        for (auto& processing_thread : processing_threads) {
            processing_thread.join();
        }
        cv::destroyAllWindows();
        videoWriter.release();
    });
    
    displayThread.join();

#else
    cv::Mat img = cv::imread("images/test5.jpg");
    if (img.empty()) {
        return EXIT_FAILURE;
    }
    imshow("Original image", img);
    cv::waitKey(0);
    std::cout << "Image size: " << img.size() << std::endl;

    std::vector<cv::Point2f> ROI_points, warp_destination_points;
    cv::Mat M, Minv;          
    // combined(img, thresholded);
    combined(img, thresholded);
    calculateWarpPoints(img, ROI_points, warp_destination_points);
    perspectiveTransform(ROI_points, warp_destination_points, M, Minv); 
    perspectiveWarp(thresholded, warped, M);

    imshow("Thresholded", thresholded);
    cv::waitKey(0);
    imshow("Warped", warped);
    cv::waitKey(0);

    cv::Mat histogram;
    calculateHistogram(warped, histogram);

    int left_peak, right_peak;
    findLaneHistogramPeaks(histogram, left_peak, right_peak);

    plotHistogram(histogram, warped);

    std::vector<double> fity;
    fity = linspace(0, warped.rows - 1, warped.rows);

    int num_of_windows = 9;
    int window_height = warped.rows / num_of_windows;

    std::vector<cv::Point> nonzero_points;
    cv::findNonZero(warped, nonzero_points);
    std::vector<double> nonzero_x, nonzero_y;

    for (const auto& point : nonzero_points) {
        nonzero_x.push_back(point.x);
        nonzero_y.push_back(point.y);
    }

    // Width of the windows +- margin
    int margin {50};
    // Minimum pixel number to recenter windows
    int minpix {25};

    std::vector<int> left_line_indices, right_line_indices;

    cv::Mat win_result_img(warped.size(), CV_8UC3, cv::Scalar(0,0,0));
    // cv::cvtColor(warped, win_result_img, cv::COLOR_GRAY2BGR);

    cv::Mat good_left_inds, good_right_inds;
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
    vector<int> leftx, lefty, rightx, righty;
    for (int idx : left_line_indices) {
        leftx.push_back(nonzero_x[idx]);
        lefty.push_back(nonzero_y[idx]);
    }
    for (int idx : right_line_indices) {
        rightx.push_back(nonzero_x[idx]);
        righty.push_back(nonzero_y[idx]);
    }

    // Fit a second order polynomial to each
    Mat left_fit = polyfit(lefty, leftx, 2);
    Mat right_fit = polyfit(righty, rightx, 2);

    // Generate x and y values for plotting
    vector<int> ploty(warped.rows);
    for (size_t i = 0; i < ploty.size(); ++i) {
        ploty[i] = static_cast<int>(i);
    }
    vector<int> left_fitx(ploty.size());
    vector<int> right_fitx(ploty.size());
    for (size_t i = 0; i < ploty.size(); ++i) {
        left_fitx[i] = left_fit.at<double>(2, 0) * ploty[i] * ploty[i] +
                       left_fit.at<double>(1, 0) * ploty[i] +
                       left_fit.at<double>(0, 0);
        right_fitx[i] = right_fit.at<double>(2, 0) * ploty[i] * ploty[i] +
                        right_fit.at<double>(1, 0) * ploty[i] +
                        right_fit.at<double>(0, 0);
    }

    // Generate black image and colour lane lines
    Mat lane_lines(warped.size(), CV_8UC3, Scalar(0, 0, 0));

    // Draw polyline on image
    vector<Point> left_polyline, right_polyline;
    for (size_t i = 0; i < ploty.size(); ++i) {
        left_polyline.emplace_back(left_fitx[i], ploty[i]);
        right_polyline.emplace_back(right_fitx[i], ploty[i]);
    }
    const Point *left_polyline_data = &left_polyline[0];
    const Point *right_polyline_data = &right_polyline[0];
    int num_points = static_cast<int>(ploty.size());
    polylines(lane_lines, &left_polyline_data, &num_points, 1, false, Scalar(0, 255, 0), 5);
    polylines(lane_lines, &right_polyline_data, &num_points, 1, false, Scalar(0, 255, 0), 5);

    // Display results
    imshow("Lane Lines", lane_lines);
    waitKey(0);

    // Convert ploty and left_fitx, right_fitx to cv::Mat for warping
    cv::Mat plotyMat = cv::Mat(ploty).reshape(1);
    cv::Mat left_fitxMat = cv::Mat(left_fitx).reshape(1);
    cv::Mat right_fitxMat = cv::Mat(right_fitx).reshape(1);

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
    cv::warpPerspective(lane_area, new_warp, Minv, img.size());

    // Blend the lane area with the original image
    cv::Mat result;
    cv::addWeighted(img, 1, new_warp, 0.3, 0, result);

    // Display the final image with the filled lane area
    cv::imshow("Lane Area on Original Image", result);
    cv::waitKey(0);

    cv::destroyAllWindows();

#endif
    return 0;
}