#include "common.h"
#include "cudaUtility.h"
#include "mathFunctions.h"
#include "pluginImplement.h"
#include "tensorNet.h"
#include "loadImage.h"
#include <chrono>
#include <thread>
#include "Timer.h"
#include "util.h"

const char* model  = "../../model/MobileNetSSD_deploy_iplugin.prototxt";
const char* weight = "../../model/MobileNetSSD_deploy.caffemodel";

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "detection_out";

static const uint32_t BATCH_SIZE = 1;
const int CAMID = 1;

const int HEIGHT = 300;
const int WIDTH  = 300;

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


int main(int argc, char *argv[])
{
    std::vector<std::string> output_vector = {OUTPUT_BLOB_NAME};

    TensorNet tensorNet;
    tensorNet.LoadNetwork(model, weight, INPUT_BLOB_NAME, output_vector, BATCH_SIZE);

    DimsCHW dimsData = tensorNet.getTensorDims(INPUT_BLOB_NAME);
    DimsCHW dimsOut  = tensorNet.getTensorDims(OUTPUT_BLOB_NAME);

    cv::VideoCapture cap;

    if (!cap.open(CAMID))
    {
        std::cout << "cannot open camera " << CAMID << std::endl;
        return -1;
    }

    float* output = allocateMemory(dimsOut, (char*)"output blob");
    std::cout << "allocate output" << std::endl;
    
    
    cv::Mat image, srcImg; // camera frame, displayed clone
    void* imgCUDA;

    const size_t size = WIDTH * HEIGHT * sizeof(float3);
    void* imgData = malloc(size); // normalized image

    if (CUDA_FAILED(cudaMalloc(&imgCUDA, size)))
    {
        cout <<"Cuda Memory allocation error occured." << endl;
        return -1;
    }

    Timer timer;

    while(1)
    {
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

        cv::resize(image, image, cv::Size(WIDTH, HEIGHT));

        
        std::cout << "IMAGE rows, cols" << image.rows << " " << image.cols << std::endl;
        
        //uint8_t CV_8U
        unsigned char *h_a;

        /* clone mat to gpu
            cast to float
            cudanormalize
        
            cudaHostAlloc
            cudaFreeHost
        */


        loadImg(image, HEIGHT, WIDTH, (float*)imgData, make_float3(127.5,127.5,127.5), 0.007843);

        // imgData normalized float image
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
    cudaFree(output);
    tensorNet.destroy();
    return 0;
}
