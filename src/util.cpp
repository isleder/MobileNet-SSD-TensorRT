
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include "util.h"

void loadImg(cv::Mat &input, 
             int re_width, 
             int re_height, 
             float *data_unifrom, 
             const float3 mean,
             const float scale)
{
    int line_offset;
    int offset_g;
    int offset_r;
    cv::Mat dst;

    unsigned char *line = NULL;
    float *unifrom_data = data_unifrom;

    cv::resize(input, dst, cv::Size( re_width, re_height ), (0.0), (0.0), cv::INTER_LINEAR);

    offset_g = re_width * re_height;
    offset_r = re_width * re_height * 2;

    for(int i = 0; i < re_height; ++i)
    {
        line = dst.ptr< unsigned char >(i);
        line_offset = i * re_width;

        for(int j = 0; j < re_width; ++j)
        {
            // b
            unifrom_data[line_offset + j] = ((float)(line[j * 3] - mean.x) * scale);
            // g
            unifrom_data[offset_g + line_offset + j] = ((float)(line[j * 3 + 1] - mean.y) * scale);
            // r
            unifrom_data[offset_r + line_offset + j] = ((float)(line[j * 3 + 2] - mean.z) * scale);
        }
    }
}

//thread read video
/*
void readPicture()
{
    cv::VideoCapture cap("../../testVideo/test.avi");
    cv::Mat image;

    while(cap.isOpened())
    {
        cap >> image;
        imageBuffer->add(image);
    }
}
*/
/*
int openCamera(int camid)
{
    cv::VideoCapture cap;
    cv::Mat image;

    if (!cap.open(camid))
    {
        return -1;
    }

    while(!quit)
    {

        if (!cap.read(image)) break;
        if (image.empty()) break;

        cv::waitKey(1);
    }
}
*/