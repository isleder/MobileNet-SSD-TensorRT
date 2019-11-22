#include <chrono>

class Timer {

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_;

public:
    void tic() 
    {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double toc() 
    {
        end_ = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_ - start_).count();
    }
};
