// src/Main.cpp
#include <iostream>
#include <stdlib.h>
#include <string>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include "Utils.hpp"

namespace fs = std::filesystem;

#define USE_THREADS

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
    std::atomic<bool> process(true);

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
        int processed_frame_count = 0;
        double total_processing_time = 0.0;

        while(process || !frame_queue.empty()) {
            cv::Mat frame, output;

            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cond_var.wait(lock, [&]() {
                return !frame_queue.empty() || !process;
            });

            if (!frame_queue.empty()) {
                frame = frame_queue.front();
                frame_queue.pop();
                lock.unlock();
            }

            if (!frame.empty()) {
                auto process_start = std::chrono::steady_clock::now();

                pipeline(frame, output);

                auto process_end = std::chrono::steady_clock::now();
                std::chrono::duration<double> process_duration = process_end - process_start;
                processed_frame_count++;
                total_processing_time += process_duration.count();
                // Adjust waitKey delay to account for processing time, ensuring a minimal delay
                int delay = std::max(1, 30 - static_cast<int>(process_duration.count() * 1000));

                // videoWriter.write(output);
                imshow("Final frame:", output);

                if (cv::waitKey(delay) == 27) {
                    process = false;
                }

                if (processed_frame_count % 30 == 0) {
                    double fps_processing = processed_frame_count / total_processing_time;
                    std::cout << "Processing FPS: " << fps_processing << std::endl;

                    processed_frame_count = 0;
                    total_processing_time = 0.0;
                }
            }
        }
    });

    readThread.join();
    processThread.join();
    videoWriter.release();
    cv::destroyAllWindows();

#else
    std::string images_folder = "../images";

    for (const auto & entry : std::filesystem::directory_iterator(images_folder)) {
        std::string file_path = entry.path().string();
        std::string file_name = extractFilename(file_path);

        std::filesystem::create_directories("../images_results/" + file_name);

        image_results_path = "../images_results/" + file_name + "/";
        image_results_filename = "_" + file_name;

        cv::Mat img = cv::imread(file_path);
        if (img.empty()) {
            return EXIT_FAILURE;
        }

        cv::Mat output;
        pipeline(img, output);

        cv::destroyAllWindows();
    }

#endif
    return 0;
}