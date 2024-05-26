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
#include "QueueFPS.hpp"
#include "Utils.hpp"

using namespace std;
using namespace cv;

// #define USE_MULTIPLE_THREADS
#define USE_2_THREADS
// #define NO_THREADS
// #define TEST_ON_IMAGE
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

    cv::VideoWriter videoWriter;
    double fps = 30.0;
    cv::Size frame_size(1280, 720);
    videoWriter.open("output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frame_size);

    if (!videoWriter.isOpened()) {
        std::cerr << "Error: Could not open the output video file for write\n";
    }

    std::queue<cv::Mat> frame_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cond_var;
    bool process = true;

    double fps_show = 0.0;
    int frame_count = 0;
    auto start = std::chrono::steady_clock::now();

    std::thread readThread([&]() {
        cv::Mat frame;
        while (process) {
            if (video_capture.read(frame)) {
                std::unique_lock<std::mutex> lock(queue_mutex);
                frame_queue.push(frame);
                lock.unlock();
                queue_cond_var.notify_one();
            }
            else {
                process = false;
            }
        }
    });

    std::thread processThread([&]() {
        while(process || !frame_queue.empty()) {
            cv::Mat frame, output;

            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cond_var.wait(lock, [&]() {
                return !frame_queue.empty() || !process;
            });

            if (!frame_queue.empty()) {
                frame = frame_queue.front();
                frame_queue.pop();
            }

            if (!frame.empty()) {
                auto processStart = std::chrono::steady_clock::now();

                pipeline(frame, output);
                getFPS(start, frame_count, fps_show);

                std::cout << std::to_string(fps) << std::endl;

                auto processEnd = std::chrono::steady_clock::now();
                std::chrono::duration<double> processDuration = processEnd - processStart;
                // Adjust waitKey delay to account for processing time, ensuring a minimal delay
                int delay = std::max(1, 30 - static_cast<int>(processDuration.count() * 1000));

                imshow("Final frame:", output);

                if (cv::waitKey(delay) == 27) {
                    process = false;
                }
            }
        }
    });

    readThread.join();
    processThread.join();
    cv::destroyAllWindows();

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
    // detectEdges(img, thresholded);
    canny(img, thresholded);
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

    plotLinesOnWarped(warped, ploty, left_fitx, right_fitx);

    plotMarkedLane(img, warped, Minv, ploty, left_fitx, right_fitx);

    cv::destroyAllWindows();

#endif
    return 0;
}