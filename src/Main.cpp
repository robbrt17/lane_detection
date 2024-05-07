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

// #define USE_MULTIPLE_THREADS
// #define USE_2_THREADS
// #define NO_THREADS
#define TEST_ON_IMAGE

int thread_number = 3;

cv::Mat thresholded, abs_sobel;
cv::Mat warped, unwarped;   

int main() 
{
    // Get test video
    cv::VideoCapture video_capture("./test_videos/project_video_lower_res.mp4");

    // If it can't be opened, return error message and exit
    if (video_capture.isOpened() == false) 
    {
        std::cout << "THE VIDEO FILE COULD NOT BE OPENED OR WRONG FORMAT!" << std::endl;
        std::cin.get();
        return -1;
    }

    // Display number of FPS
    std::cout << "Video frames per second: " << video_capture.get(cv::CAP_PROP_FPS) << std::endl;

#ifdef USE_MULTIPLE_THREADS
    bool process = true;

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

    QueueFPS<cv::Mat> thresholdedFramesQueue;
    std::thread thresholdingThread([&](){
        while (process)
        {
            cv::Mat frame;

            if (!framesQueue.empty())
            {
                frame = framesQueue.get();
            }

            if (!frame.empty())
            {
                threshold.canny(frame, thresholded); 

                thresholdedFramesQueue.push(thresholded);
            }
        }
    });

    QueueFPS<cv::Mat> processedFramesQueue;
    std::thread processingThread([&](){
        while (process)
        {
            cv::Mat frame;

            if (!thresholdedFramesQueue.empty())
            {
                frame = thresholdedFramesQueue.get();
            }

            if (!frame.empty())
            {
                std::vector<cv::Point2f> ROI_points, warp_destination_points;
                cv::Mat M, Minv;          
                // threshold.combined(frame, thresholded); 
                perspective.calculateWarpPoints(frame, ROI_points, warp_destination_points);
                perspective.perspectiveTransform(ROI_points, warp_destination_points, M, Minv);
                perspective.perspectiveWarp(frame, warped, M);

                processedFramesQueue.push(warped);
            }
        }
    });

    while (true)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        cv::Mat frame;
        if(!processedFramesQueue.empty()){
            frame = processedFramesQueue.get();
            std::cout << "Frames Queue FPS:" << framesQueue.getFPS() << std::endl;
            std::cout << "Processed Frames Queue FPS:" << processedFramesQueue.getFPS() << std::endl;
            std::cout << "==============================" << std::endl;

            imshow("test", frame);
            auto end_time = std::chrono::high_resolution_clock::now();
            int elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            // std::cout << elapsed_time << std::endl;

            // std::cout << "FPS: " << (video_capture.get(cv::CAP_PROP_FRAME_COUNT) / elapsed_time) << std::endl;
        }

        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }

    process = false;
    framesThread.join();
    thresholdingThread.join();
    processingThread.join();
    cv::destroyAllWindows();

#elif defined(USE_2_THREADS)

    bool process = true;

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
    std::thread processingThread([&](){
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
                threshold.canny(frame, thresholded);
                perspective.calculateWarpPoints(thresholded, ROI_points, warp_destination_points);
                perspective.perspectiveTransform(ROI_points, warp_destination_points, M, Minv);
                perspective.perspectiveWarp(thresholded, warped, M);
                processedFramesQueue.push(warped);
            }
        }
    });

    while (true)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        cv::Mat frame;
        if(!processedFramesQueue.empty()){
            frame = processedFramesQueue.get();
            std::cout << "Frames Queue FPS:" << framesQueue.getFPS() << std::endl;
            std::cout << "Processed Frames Queue FPS:" << processedFramesQueue.getFPS() << std::endl;
            std::cout << "==============================" << std::endl;

            imshow("test", frame);
        }

        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }

    process = false;
    framesThread.join();
    processingThread.join();
    cv::destroyAllWindows();

#elif defined(NO_THREADS)
    for (int frame_index = 0; frame_index < (int)video_capture.get(cv::CAP_PROP_FRAME_COUNT); frame_index++)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        cv::Mat frame, resized_frame;
        if (!video_capture.read(frame))
        {
            std::cout << "blank frame grabbed" << std::endl;
            break;
        }

        // cv::resize(frame, frame, cv::Size(640, 380),0,0, cv::INTER_LINEAR);

        if (frame_index % 2 == 0)
        {
            std::vector<cv::Point2f> ROI_points, warp_destination_points;
            cv::Mat M, Minv;          
            threshold.combined(frame, thresholded);
            perspective.calculateWarpPoints(frame, ROI_points, warp_destination_points);
            perspective.perspectiveTransform(ROI_points, warp_destination_points, M, Minv); 
            perspective.perspectiveWarp(thresholded, warped, M);
            
            
            // cv::warpPerspective(warped, unwarped, Minv, warped.size(), cv::INTER_LINEAR);
            // threshold.combined(frame, thresholded);
        }

        cv::imshow("Test video", frame);

        cv::imshow("Thresholded", thresholded);
        // cv::imshow("Warped", warped);
        // cv::imshow("Unwarped", unwarped);

        auto end_time = std::chrono::high_resolution_clock::now();
        int elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << elapsed_time << std::endl;

        std::cout << "FPS: " << (video_capture.get(cv::CAP_PROP_FRAME_COUNT) / elapsed_time) << std::endl;

        int wait_time = std::max(1, elapsed_time);
        if (cv::waitKey(wait_time) == 27)
        {
            break;
        }
    }

    video_capture.release();
    cv::destroyAllWindows();

#else
    cv::Mat img = cv::imread("images/test_lower_res.jpg");
    if (img.empty()) {
        return EXIT_FAILURE;
    }
    imshow("Original image", img);
    cv::waitKey(0);

    std::vector<cv::Point2f> ROI_points, warp_destination_points;
    cv::Mat M, Minv;          
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
    std::cout << histogram << std::endl;

    int left_peak, right_peak;
    findLaneHistogramPeaks(histogram, left_peak, right_peak);

    plotHistogram(histogram, warped);

    int num_of_windows = 9;
    int window_height = warped.rows / num_of_windows;

    std::vector<cv::Point> nonzero_points;
    // cv::findNonZero(warped, nonzero_points);
    // cv::Mat nonzero_points_mat(nonzero_points);
    // cv::Mat nonzero_x = nonzero_points_mat.col(0);
    // cv::Mat nonzero_y = nonzero_points_mat.col(1);

    int margin = 100;
    int minpix = 50;

    std::vector<cv::Mat> left_lane_inds, right_lane_inds;


    // cv::Mat win_result_img(warped.size(), CV_8UC3, cv::Scalar(0,0,0));
    // cv::cvtColor(warped, win_result_img, cv::COLOR_GRAY2BGR);

    for (int window = 0; window < num_of_windows; window++) {
        int win_y_low = warped.rows - (window + 1) * window_height;
        int win_y_high = warped.rows - window * window_height;
        
        int win_xleft_low = left_peak - margin;
        int win_xleft_high = left_peak + margin;
        int win_xright_low = right_peak - margin;
        int win_xright_high = right_peak + margin;

        // cv::rectangle(win_result_img, cv::Point(win_xleft_low, win_y_low), cv::Point(win_xleft_high, win_y_high), cv::Scalar(0,255,0), 2);
        // cv::rectangle(win_result_img, cv::Point(win_xright_low, win_y_low), cv::Point(win_xright_high, win_y_high), cv::Scalar(0,255,0), 2);
    }

    // imshow("windows", win_result_img);
    // cv::waitKey(0);

    cv::destroyAllWindows();

#endif
    return 0;
}