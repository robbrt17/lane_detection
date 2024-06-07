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
#include "Utils.hpp"

// #define USE_THREADS

cv::Mat thresholded, abs_sobel;
cv::Mat warped, unwarped;   

int main() 
{
#ifdef USE_THREADS
    // Get test video
    cv::VideoCapture video_capture("/home/robebala/stuff/licenta/lane_detection/test_videos/project_video.mp4");

    // If it can't be opened, return error message and exit
    if (video_capture.isOpened() == false) 
    {
        std::cout << "THE VIDEO FILE COULD NOT BE OPENED OR WRONG FORMAT!" << std::endl;
        std::cin.get();
        return -1;
    }

    // Display number of FPS
    std::cout << "Video frames per second: " << video_capture.get(cv::CAP_PROP_FPS) << std::endl;
    
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

                auto processEnd = std::chrono::steady_clock::now();
                std::chrono::duration<double> processDuration = processEnd - processStart;
                // Adjust waitKey delay to account for processing time, ensuring a minimal delay
                int delay = std::max(1, 30 - static_cast<int>(processDuration.count() * 1000));

                videoWriter.write(output);
                // imshow("Final frame:", output);

                if (cv::waitKey(delay) == 27) {
                    process = false;
                }
            }
        }
    });

    readThread.join();
    processThread.join();
    videoWriter.release();
    cv::destroyAllWindows();

#else
    cv::Mat img = cv::imread("/home/robebala/stuff/licenta/lane_detection/images/test5.jpg");
    if (img.empty()) {
        return EXIT_FAILURE;
    }
    cv::Mat output;
    imshow("Original image", img);
    cv::waitKey(0);
    std::cout << "Image size: " << img.size() << std::endl;

    pipeline(img, output);

    cv::destroyAllWindows();

#endif
    return 0;
}