#include <iostream>
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
#include "LaneDetector.hpp"
#include "QueueFPS.hpp"

#define USE_THREADS

int main() 
{
    Threshold threshold;
    PerspectiveTransform perspective;

    cv::Mat image = cv::imread("./images/test2.jpg", 1);
    std::cout << "Number of channels: " << image.channels();

    cv::namedWindow("test", cv::WINDOW_NORMAL);

    // Get test video
    cv::VideoCapture video_capture("./test_videos/Video_output.mp4");

    // If it can't be opened, return error message and exit
    if (video_capture.isOpened() == false) 
    {
        std::cout << "THE VIDEO FILE COULD NOT BE OPENED OR WRONG FORMAT!" << std::endl;
        std::cin.get();
        return -1;
    }

    // Display number of FPS
    std::cout << "Video frames per second: " << video_capture.get(cv::CAP_PROP_FPS) << std::endl;
    cv::Mat thresholded, abs_sobel, hls[3];
    cv::Mat warped, unwarped;

    int counter = 0;

#ifdef USE_THREADS
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
    QueueFPS<std::vector<cv::Mat>> predictionsQueue;
    std::thread processingThread([&](){
        std::queue<cv::AsyncArray> futureOutputs;
        cv::Mat blob;

        while (process)
        {
            cv::Mat frame;

            if (!framesQueue.empty())
            {
                frame = framesQueue.get();
                counter++;
            }

            if (!frame.empty())
            {
                std::vector<cv::Point2f> ROI_points, warp_destination_points;
                cv::Mat M, Minv;          

                perspective.calculateWarpPoints(frame, ROI_points, warp_destination_points, 720, 425);
                perspective.perspectiveTransform(ROI_points, warp_destination_points, M, Minv); 
                threshold.canny(frame, thresholded);
                perspective.perspectiveWarp(thresholded, warped, M);
                // cv::warpPerspective(warped, unwarped, Minv, warped.size(), cv::INTER_LINEAR);
                // threshold.combined(frame, thresholded);
                processedFramesQueue.push(frame);
                // std::cout << framesQueue.size() << std::endl;
            }
        }
    });

    while (1)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        cv::Mat frame;
        if(!processedFramesQueue.empty()){
            frame = processedFramesQueue.get();
            // std::cout << "Frames Queue FPS:" << framesQueue.getFPS() << std::endl;
            // std::cout << "Processed Frames Queue FPS:" << processedFramesQueue.getFPS() << std::endl;
            // std::cout << "==============================" << std::endl;


            imshow("test", frame);
            auto end_time = std::chrono::high_resolution_clock::now();
            int elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            // std::cout << elapsed_time << std::endl;

            std::cout << "FPS: " << (video_capture.get(cv::CAP_PROP_FRAME_COUNT) / elapsed_time) << std::endl;
        }

        

        if (cv::waitKey(1) == 27)
        {
            break;
        }
    }

    process = false;
    framesThread.join();
    processingThread.join();

    std::cout << processedFramesQueue.size() << std::endl;

#else
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

            // perspective.calculateWarpPoints(frame, ROI_points, warp_destination_points, 720, 425);
            // perspective.perspectiveTransform(ROI_points, warp_destination_points, M, Minv); 
            threshold.canny(frame, thresholded);
            // perspective.perspectiveWarp(thresholded, warped, M);
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
#endif
    return 0;
}