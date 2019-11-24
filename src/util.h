#ifndef UTIL_H
#define UTIL_H

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cuda.h>
#include <cuda_runtime.h>



void loadImg(cv::Mat &input, 
             int re_width, 
             int re_height, 
             float *data_unifrom, 
             const float3 mean,
             const float scale);
#endif