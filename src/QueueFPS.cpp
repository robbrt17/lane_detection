#include "QueueFPS.hpp"

template <typename T>
QueueFPS<T>::QueueFPS() {
    counter = 0;
}

template <typename T>
void QueueFPS<T>::push(const T& entry) {
    std::lock_guard<std::mutex> lock(mutex);
    std::queue<T>::push(entry);
    counter += 1;

    if (counter == 1)
    {
        tm.reset();
        tm.start();
    }
}

template <typename T>
T QueueFPS<T>::get() {
    std::lock_guard<std::mutex> lock(mutex);

    T entry = this->front();
    this->pop();

    return entry;
}

template <typename T>
float QueueFPS<T>::getFPS() {
    tm.stop();
    double fps = counter / tm.getTimeSec();
    tm.start();

    return static_cast<float>(fps);
}

template <typename T>
void QueueFPS<T>::clear() {
    std::lock_guard<std::mutex> lock(mutex);
    while (!this->empty())
    {
        this->pop();
    }  
}