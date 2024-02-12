#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <queue>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <filesystem>
#include "CameraCalibration.hpp"
#include "PerspectiveTransform.hpp"
#include "Threshold.hpp"
#include "LaneDetector.hpp"

int main() 
{
    Threshold threshold;
    PerspectiveTransform perspective;

    cv::Mat image = cv::imread("./images/test2.jpg", 1);
    std::cout << "Number of channels: " << image.channels();

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
    std::cout << "Frames per second: " << video_capture.get(cv::CAP_PROP_FPS) << std::endl;
cv::Mat thresholded, abs_sobel, hls[3];
    for (int frame_index = 0; frame_index < (int)video_capture.get(cv::CAP_PROP_FRAME_COUNT); frame_index++)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        cv::Mat frame;
        if (!video_capture.read(frame))
        {
            std::cout << "blank frame grabbed" << std::endl;
            break;
        }

        cv::resize(frame, frame, cv::Size(640, 360));

        
        std::vector<cv::Point2f> ROI_points, warp_destination_points;
        cv::Mat M, Minv;
        cv::Mat warped, unwarped;

        // cv::resize(frame, frame, cv::Size(640, 480), cv::INTER_LINEAR);

        // std::cout << frame.size() << std::endl;

        if (frame_index % 2 == 0)
        {
            // perspective.calculateWarpPoints(frame, ROI_points, warp_destination_points, 720, 425);
            // perspective.perspectiveTransform(ROI_points, warp_destination_points, M, Minv); 
            threshold.canny(frame, thresholded);
            // perspective.perspectiveWarp(frame, warped, M);
            // cv::warpPerspective(warped, unwarped, Minv, warped.size(), cv::INTER_LINEAR);
            // threshold.combined(frame, thresholded);
        }

        // std::thread worker1(&PerspectiveTransform::calculateWarpPoints, &perspective, frame, ROI_points, warp_destination_points, 720, 425);
        // std::thread worker2(&PerspectiveTransform::perspectiveTransform, &perspective, ROI_points, warp_destination_points, M, Minv);
        // std::thread worker3(&PerspectiveTransform::perspectiveWarp, &perspective, frame, warped, M);
        
        // worker1.join();
        // worker2.join();
        // worker3.join();

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


    // cv::Mat thresholded_image, abs_sobel, hls[3];
    // std::vector<cv::Point2f> ROI_points, warp_destination_points;
    // cv::Mat M, Minv;
    // cv::Mat warped_image, unwarped_image;
    
    // perspective.calculateWarpPoints(image, ROI_points, warp_destination_points, 720, 425);
    // perspective.perspectiveTransform(ROI_points, warp_destination_points, M, Minv); 
    // threshold.combined(image, thresholded_image);
    // perspective.perspectiveWarp(thresholded_image, warped_image, M);
    // cv::warpPerspective(warped_image, unwarped_image, Minv, warped_image.size(), cv::INTER_LINEAR);

    // cv::imshow("Initial image", image);
    // cv::imshow("Thresholded image", thresholded_image);
    // cv::imshow("Warped image", warped_image);
    // cv::imshow("Unwarped image", unwarped_image);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    return 0;
}