#ifndef __QUEUEFPS_H__
#define __QUEUEFPS_H__

#include <iostream>
#include <sstream>
#include <fstream>
#include <thread>
#include <queue>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>

template <typename T>
class QueueFPS : public std::queue<T> {
    public:
        

        QueueFPS();

        void push(const T& entry);
        T get();
        float getFPS();
        void clear();

    private:
        unsigned int counter;
        cv::TickMeter tm;
        std::mutex mutex;
};

// template <typename T>
// class QueueFPS : public std::queue<T>
// {
// public:
//     QueueFPS()
//     {
//         counter = 0;
//     }

//     void push(const T& entry)
//     {
//         std::lock_guard<std::mutex> lock(mutex);
//         std::queue<T>::push(entry);
//         counter += 1;

//         if (counter == 1)
//         {
//             tm.reset();
//             tm.start();
//         }
//     }    

//     T get()
//     {
//         std::lock_guard<std::mutex> lock(mutex);
//         T entry = this->front();
//         this->pop();
//         return entry;
//     }

//     float getFPS()
//     {
//         tm.stop();
//         double fps = counter / tm.getTimeSec();
//         tm.start();
//         return static_cast<float>(fps);
//     }

//     void clear()
//     {
//         std::lock_guard<std::mutex> lock(mutex);
//         while (!this->empty())
//         {
//             this->pop();
//         }
//     }

//     unsigned int counter;

// private:
//     cv::TickMeter tm;
//     std::mutex mutex;
// };

#endif