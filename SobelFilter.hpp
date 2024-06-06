/**********************************************************************
Copyright �2014 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

�   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
�   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#ifndef SOBEL_FILTER_H_
#define SOBEL_FILTER_H_

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <thread>
#include <mutex>
#include <atomic>

#include <opencv2/opencv.hpp>

#include "CLUtil.hpp"
#include "SDKBitMap.hpp"

using namespace appsdk;

#define SAMPLE_VERSION "AMD-APP-SDK-v2.9-1.599.2"

#define INPUT_IMAGE "SobelFilter_Input.bmp"
#define OUTPUT_IMAGE "SobelFilter_Output.bmp"

#define GROUP_SIZE 128

/**
* SobelFilter
* Class implements OpenCL Sobel Filter sample
*/

class SobelFilter
{
        cv::VideoCapture cap_;
        cl_double setupTime;                /**< time taken to setup OpenCL resources and building kernel */
        cl_double kernelTime;               /**< time taken to run kernel and read result back */
        cl_uchar4* inputImageData;          /**< Input bitmap data to device */
        cl_uchar4* outputImageData;         /**< Output from device */
        cl_context context;                 /**< CL context */
        cl_device_id *devices;              /**< CL device list */
        cl_mem inputImageBuffer;            /**< CL memory buffer for input Image*/
        cl_mem outputImageBuffer;           /**< CL memory buffer for Output Image*/
        cl_uchar* verificationOutput;       /**< Output array for reference implementation */
        cl_command_queue commandQueue;      /**< CL command queue */
        cl_program program;                 /**< CL program  */
        cl_kernel kernel;                   /**< CL kernel */
        SDKBitMap inputBitmap;   /**< Bitmap class object */
        uchar4* pixelData;       /**< Pointer to image data */
        cl_uint pixelSize;                  /**< Size of a pixel in BMP format> */
        cl_uint width;                      /**< Width of image */
        cl_uint height;                     /**< Height of image */
        cl_bool byteRWSupport;
        size_t kernelWorkGroupSize;         /**< Group Size returned by kernel */
        size_t blockSizeX;                  /**< Work-group size in x-direction */
        size_t blockSizeY;                  /**< Work-group size in y-direction */
        int iterations;                     /**< Number of iterations for kernel execution */
        SDKDeviceInfo
        deviceInfo;                       /**< Structure to store device information*/
        KernelWorkGroupInfo
        kernelInfo;         /**< Structure to store kernel related info */

        std::atomic<bool> stopCamThread{false};
        std::atomic<bool> stopKernelThread{false};
        std::atomic<bool> camReady{false};
        std::atomic<bool> frameTaken{false};
        std::atomic<int> key{0};
        std::atomic<int> lostFrames{0};
        std::atomic<bool> inputImageReady{false};
        std::atomic<bool> outputImageReady{false};
        std::mutex inputImageMutex;
        std::thread camThread;
        std::thread kernelThread;
        std::chrono::time_point<std::chrono::steady_clock> lastFrameTime;
        std::chrono::time_point<std::chrono::steady_clock> startTime;
        std::string kernel_f;
        int camera_num;
        std::atomic<int> no_frames{0};
        std::chrono::duration<double> frameIntervalAll{0};
        std::chrono::duration<double> kernelProcessingTimeAll{0};
        std::atomic<int> no_kernel_calls{0};
        cl_uchar4* inputImageToSave;
        cl_uchar4* outputImageToSave;

        SDKTimer    *sampleTimer;      /**< SDKTimer object */

    public:

        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */

        /**
        * Read bitmap image and allocate host memory
        * @param inputImageName name of the input file
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        void readInputImage();

        /**
        * Write to an image file
        * @param outputImageName name of the output file
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int writeOutputImage(std::string outputImageName);

        /**
        * Constructor
        * Initialize member variables
        */
        SobelFilter()
            : inputImageData(NULL),
              outputImageData(NULL),
              verificationOutput(NULL),
              byteRWSupport(true)
        {
            camera_num = 0;
            sampleArgs = new CLCommandArgs();
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
            pixelSize = sizeof(uchar4);
            pixelData = NULL;
            blockSizeX = GROUP_SIZE;
            blockSizeY = 1;
            iterations = 1;
            kernel_f = "SobelFilter_Kernels.cl";
        }

        ~SobelFilter()
        {
            cv::destroyAllWindows();
            if (cap_.isOpened())
            {
                cap_.release();
            }
            if (camThread.joinable())
            {
                camThread.join();
            }
            if (kernelThread.joinable())
            {
                kernelThread.join();
            }
        }

        cv::Mat camera();

        int writeInputImage(std::string inputImageName);

        void readKeys();

        /**
        * Allocate image memory and Load bitmap file
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupSobelFilter();

        /**
        * OpenCL related initialisations.
        * Set up Context, Device list, Command Queue, Memory buffers
        * Build CL kernel program executable
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupCL();

        /**
        * Set values for kernels' arguments, enqueue calls to the kernels
        * on to the command queue, wait till end of kernel execution.
        * Get kernel start and end time if timing is enabled
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int runCLKernels();

        /**
        * Reference CPU implementation of Binomial Option
        * for performance comparison
        */
        void sobelFilterCPUReference();

        /**
        * Override from SDKSample. Print sample stats.
        */
        void printStats();

        /**
        * Override from SDKSample. Initialize
        * command line parser, add custom options
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int initialize();

        /**
         * Override from SDKSample, Generate binary image of given kernel
         * and exit application
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int genBinaryImage();

        /**
        * Override from SDKSample, adjust width and height
        * of execution domain, perform all sample setup
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setup();

        /**
        * Override from SDKSample
        * Run OpenCL Sobel Filter
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int run();

        /**
        * Override from SDKSample
        * Cleanup memory allocations
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int cleanup();

        /**
        * Override from SDKSample
        * Verify against reference implementation
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int verifyResults();

        void stop();
};

#endif // SOBEL_FILTER_H_
