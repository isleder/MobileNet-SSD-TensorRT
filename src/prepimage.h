#ifndef PREPIMAGE_H
#define PREPIMAGE_H

#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>

//void prepimage(const cv::Mat& input, cv::Mat& output);

cudaError_t prepimage(const cv::Mat& input, float3* output);




#endif
