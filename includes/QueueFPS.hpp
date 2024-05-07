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

#endif