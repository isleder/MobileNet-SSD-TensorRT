#include "common.h"
#include "cudaUtility.h"
#include "mathFunctions.h"
#include "pluginImplement.h"
#include "tensorNet.h"
#include "loadImage.h"
#include <chrono>
#include <thread>
#include "Timer.h"

const char* model  = "../../model/MobileNetSSD_deploy_iplugin.prototxt";
const char* weight = "../../model/MobileNetSSD_deploy.caffemodel";

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "detection_out";

static const uint32_t BATCH_SIZE = 1;
const int CAMID = 1;



/* *
 * @TODO: unifiedMemory is used here under -> ( cudaMallocManaged )
 * */
float* allocateMemory(DimsCHW dims, char* info)
{
    float* ptr;
    size_t size;

    std::cout << "Allocate memory: " << info << std::endl;
    size = BATCH_SIZE * dims.c() * dims.h() * dims.w();

    assert(!cudaMallocManaged(&ptr, size*sizeof(float)));
    return ptr;
}


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

int main(int argc, char *argv[])
{
    std::vector<std::string> output_vector = {OUTPUT_BLOB_NAME};

    TensorNet tensorNet;
    tensorNet.LoadNetwork(model,weight,INPUT_BLOB_NAME, output_vector,BATCH_SIZE);

    DimsCHW dimsData = tensorNet.getTensorDims(INPUT_BLOB_NAME);
    DimsCHW dimsOut  = tensorNet.getTensorDims(OUTPUT_BLOB_NAME);

    //float* data = allocateMemory(dimsData, (char*)"input blob");
    //std::cout << "allocate data" << std::endl;

    float* output = allocateMemory(dimsOut, (char*)"output blob");
    std::cout << "allocate output" << std::endl;
    
    int height = 300;
    int width  = 300;

    //cv::Mat frame, srcImg;
    cv::Mat srcImg;

    void* imgCPU;
    void* imgCUDA;
    Timer timer;

    //std::thread readTread(readPicture);
    //readTread.detach();
    //
    //
    cv::VideoCapture cap;
    cv::Mat image;

    if (!cap.open(CAMID))
    {
        std::cout << "cannot open camera " << CAMID << std::endl;
        return -1;
    }

    const size_t size = width * height * sizeof(float3);
    void* imgData = malloc(size);
    //memset(imgData, 0, size);
    
    while(1)
    {
        //imageBuffer->consume(frame);

        if (!cap.read(image)) 
        {
            std::cout << "cannot read camera image\n";
            break;
        }

        if (image.empty())
        {            
            std::cout << "camera image empty\n";
            break;
        }

        srcImg = image.clone();
        cv::resize(image, image, cv::Size(300,300));
        
        if (CUDA_FAILED(cudaMalloc(&imgCUDA, size)))
        {
            cout <<"Cuda Memory allocation error occured." << endl;
            break;
        }

        loadImg(image, height, width, (float*)imgData, make_float3(127.5,127.5,127.5), 0.007843);

        cudaMemcpyAsync(imgCUDA, imgData, size, cudaMemcpyHostToDevice);


        void* buffers[] = { imgCUDA, output };

        timer.tic();
        tensorNet.imageInference(buffers, output_vector.size() + 1, BATCH_SIZE);
        double msTime = timer.toc();
        std::cout << msTime << std::endl;

        vector<vector<float> > detections;

        for (int k=0; k < 100; k++)
        {
            if (output[7 * k + 1] == -1) break;

            float classIndex = output[7*k+1];
            float confidence = output[7*k+2];
            float xmin = output[7*k + 3];
            float ymin = output[7*k + 4];
            float xmax = output[7*k + 5];
            float ymax = output[7*k + 6];

            std::cout << classIndex << " , " 
                      << confidence << " , "  
                      << xmin << " , " 
                      << ymin<< " , " 
                      << xmax<< " , " 
                      << ymax << std::endl;

            int x1 = static_cast<int>(xmin * srcImg.cols);
            int y1 = static_cast<int>(ymin * srcImg.rows);
            int x2 = static_cast<int>(xmax * srcImg.cols);
            int y2 = static_cast<int>(ymax * srcImg.rows);

            cv::rectangle(srcImg, 
                          cv::Rect2f(cv::Point(x1,y1),
                          cv::Point(x2,y2)),
                          cv::Scalar(255,0,255),
                          1);

        }


        cv::imshow("mobileNet",srcImg);
        char c = cv::waitKey(1);
        if (c == 27) 
        {
            break;
        }
    }

    free(imgData);
    cudaFree(imgCUDA);
    cudaFreeHost(imgCPU);
    cudaFree(output);
    tensorNet.destroy();
    return 0;
}
