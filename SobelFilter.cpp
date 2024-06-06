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


#include "SobelFilter.hpp"
#include <cmath>

const char* getKernelNameFromFile(const std::string& filename)
{
    std::ifstream file(filename);
    std::string line;
    static std::string kernelName;

    if (file.is_open()) {
        while (std::getline(file, line)) 
        {
            size_t found = line.find("__kernel");
            if (found != std::string::npos)
            {
                size_t startPos = found + 14;
                size_t endPos = line.find("(", startPos);
                kernelName = line.substr(startPos, endPos - startPos);
                break;
            }
        }
        file.close();
    } else {
        std::cerr << "Unable to open file " << filename << std::endl;
    }

    return kernelName.c_str();
}

void
SobelFilter::readInputImage()
{
    cv::namedWindow("Camera Feed", cv::WINDOW_NORMAL);
    cv::Mat inputImage = camera(); // Capture an image from the camera



    cv::Mat bgraImage;
    cv::cvtColor(inputImage, bgraImage, cv::COLOR_BGR2BGRA);
    
    // Get width and height of input image
    height = (cl_uint) (bgraImage.rows);
    width = (cl_uint) (bgraImage.cols);

    // Allocate memory for input & output image data
    inputImageData = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
    // CHECK_ALLOCATION(inputImageData, "Failed to allocate memory! (inputImageData)");

    // Allocate memory for output image data
    outputImageData = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
    // CHECK_ALLOCATION(outputImageData, "Failed to allocate memory! (outputImageData)");

    // Initialize the output image data to NULL
    memset(outputImageData, 0, width * height * sizeof(cl_uchar4));
    
    // Get the pointer to pixel data from the input image
    pixelData = (uchar4*) (bgraImage.data);
    if (pixelData == NULL)
    {
        std::cout << "Failed to read pixel Data!";
        // return SDK_FAILURE;
    }

    // Copy pixel data into inputImageData
    {
        std::lock_guard<std::mutex> lock(inputImageMutex);
        memcpy(inputImageData, pixelData, width * height * sizeof(cl_uchar4));
        frameTaken = false;
    }
    
    // Allocate memory for verification output
    verificationOutput = (cl_uchar*)malloc(width * height * sizeof(cl_uchar4));
    // CHECK_ALLOCATION(verificationOutput, "verificationOutput heap allocation failed!");

    // Initialize the data to NULL
    memset(verificationOutput, 0, width * height * sizeof(cl_uchar4));

    lastFrameTime = std::chrono::steady_clock::now();
    camReady = true;
    while(!stopCamThread)
    {
        no_frames++;
        inputImage = camera(); // Capture an image from the camera
        cv::cvtColor(inputImage, bgraImage, cv::COLOR_BGR2BGRA);

        // Get the pointer to pixel data from the input image
        pixelData = (uchar4*) (bgraImage.data);
        if (pixelData == NULL)
        {
            std::cout << "Failed to read pixel Data!";
            // return SDK_FAILURE;
        }

        {
            std::lock_guard<std::mutex> lock(inputImageMutex);
            if (frameTaken)
            {
                // Copy pixel data into inputImageData
                memcpy(inputImageData, pixelData, width * height * sizeof(cl_uchar4));
                frameTaken = false;
                auto now = std::chrono::steady_clock::now();
                std::chrono::duration<double> frameInterval = now - lastFrameTime;
                lastFrameTime = now;
                frameIntervalAll += frameInterval;
                // std::cout << "Time between frames: " << frameInterval.count() << " seconds" << std::endl;
            }
            else
            {
                lostFrames++;
                // std::cout << lostFrames << std::endl;
            }
        }
    }

    // return SDK_SUCCESS;

}

int
SobelFilter::writeInputImage(std::string inputImageName)
{
    cv::Mat mat;
    {
        std::lock_guard<std::mutex> lock(inputImageMutex);
        mat = cv::Mat(height, width, CV_8UC4, inputImageToSave);
        cv::imwrite(inputImageName, mat);
    }

    return SDK_SUCCESS;
}

int
SobelFilter::writeOutputImage(std::string outputImageName)
{
    cv::Mat mat = cv::Mat(height, width, CV_8UC4, outputImageToSave);
    cv::imwrite(outputImageName, mat);

    return SDK_SUCCESS;
}

int
SobelFilter::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = kernel_f;
    binaryData.flagsStr = std::string("");
    if(sampleArgs->isComplierFlagsSpecified())
    {
        binaryData.flagsFileName = std::string(sampleArgs->flags.c_str());
    }

    binaryData.binaryName = std::string(sampleArgs->dumpBinary.c_str());
    int status = generateBinaryImage(binaryData);
    return status;
}


int
SobelFilter::setupCL()
{
    cl_int status = CL_SUCCESS;
    cl_device_type dType;

    if(sampleArgs->deviceType.compare("cpu") == 0)
    {
        dType = CL_DEVICE_TYPE_CPU;
    }
    else //deviceType = "gpu"
    {
        dType = CL_DEVICE_TYPE_GPU;
        if(sampleArgs->isThereGPU() == false)
        {
            std::cout << "GPU not found. Falling back to CPU device" << std::endl;
            dType = CL_DEVICE_TYPE_CPU;
        }
    }

    /*
     * Have a look at the available platforms and pick either
     * the AMD one if available or a reasonable default.
     */
    cl_platform_id platform = NULL;
    int retValue = getPlatform(platform, sampleArgs->platformId,
                               sampleArgs->isPlatformEnabled());
    CHECK_ERROR(retValue, SDK_SUCCESS, "getPlatform() failed");

    // Display available devices.
    retValue = displayDevices(platform, dType);
    CHECK_ERROR(retValue, SDK_SUCCESS, "displayDevices() failed");


    // If we could find our platform, use it. Otherwise use just available platform.
    cl_context_properties cps[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };

    context = clCreateContextFromType(
                  cps,
                  dType,
                  NULL,
                  NULL,
                  &status);
    CHECK_OPENCL_ERROR( status, "clCreateContextFromType failed.");

    // getting device on which to run the sample
    status = getDevices(context,&devices,sampleArgs->deviceId,
                        sampleArgs->isDeviceIdEnabled());
    CHECK_ERROR(status, SDK_SUCCESS, "getDevices() failed");

    {
        // The block is to move the declaration of prop closer to its use
        cl_command_queue_properties prop = 0;
        commandQueue = clCreateCommandQueue(
                           context,
                           devices[sampleArgs->deviceId],
                           prop,
                           &status);
        CHECK_OPENCL_ERROR( status, "clCreateCommandQueue failed.");
    }

    //Set device info of given cl_device_id
    retValue = deviceInfo.setDeviceInfo(devices[sampleArgs->deviceId]);
    CHECK_ERROR(retValue, 0, "SDKDeviceInfo::setDeviceInfo() failed");


    // Create and initialize memory objects

    // Set Presistent memory only for AMD platform
    cl_mem_flags inMemFlags = CL_MEM_READ_ONLY;
    if(sampleArgs->isAmdPlatform())
    {
        inMemFlags |= CL_MEM_USE_PERSISTENT_MEM_AMD;
    }

    // Create memory object for input Image
    inputImageBuffer = clCreateBuffer(
                           context,
                           inMemFlags,
                           width * height * pixelSize,
                           0,
                           &status);
    
    CHECK_OPENCL_ERROR(status, "clCreateBuffer failed. (inputImageBuffer)");

    // Create memory objects for output Image
    outputImageBuffer = clCreateBuffer(context,
                                       CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                                       width * height * pixelSize,
                                       outputImageData,
                                       &status);
    CHECK_OPENCL_ERROR(status, "clCreateBuffer failed. (outputImageBuffer)");

    // create a CL program using the kernel source
    buildProgramData buildData;
    buildData.kernelName = kernel_f;
    buildData.devices = devices;
    buildData.deviceId = sampleArgs->deviceId;
    buildData.flagsStr = std::string("");
    if(sampleArgs->isLoadBinaryEnabled())
    {
        buildData.binaryName = std::string(sampleArgs->loadBinary.c_str());
    }

    if(sampleArgs->isComplierFlagsSpecified())
    {
        buildData.flagsFileName = std::string(sampleArgs->flags.c_str());
    }

    retValue = buildOpenCLProgram(program, context, buildData);
    CHECK_ERROR(retValue, 0, "buildOpenCLProgram() failed");

    // get a kernel object handle for a kernel with the given name
    const char* kernel_name = getKernelNameFromFile(kernel_f);
    
    kernel = clCreateKernel(
                 program,
                 kernel_name,
                 &status);
    CHECK_OPENCL_ERROR(status, "clCreateKernel failed.");

    status = kernelInfo.setKernelWorkGroupInfo(kernel,
             devices[sampleArgs->deviceId]);
    CHECK_ERROR(status, SDK_SUCCESS,"kernelInfo.setKernelWorkGroupInfo() failed");


    if((blockSizeX * blockSizeY) > kernelInfo.kernelWorkGroupSize)
    {
        if(!sampleArgs->quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified : "
                      << blockSizeX * blockSizeY << std::endl;
            std::cout << "Max Group Size supported on the kernel : "
                      << kernelWorkGroupSize << std::endl;
            std::cout << "Falling back to " << kernelInfo.kernelWorkGroupSize << std::endl;
        }

        // Three possible cases
        if(blockSizeX > kernelInfo.kernelWorkGroupSize)
        {
            blockSizeX = kernelInfo.kernelWorkGroupSize;
            blockSizeY = 1;
        }
    }
    return SDK_SUCCESS;
}

int
SobelFilter::runCLKernels()
{
    while (!stopKernelThread)
    {
        cl_int status;
        {
            std::lock_guard<std::mutex> lock(inputImageMutex);
            if (frameTaken)
            {
                continue;
            }
            // inputImageReady = false;
            outputImageReady = false;
            inputImageToSave = inputImageData;
            startTime = std::chrono::steady_clock::now();
            // Set input data
            cl_event writeEvt;
            status = clEnqueueWriteBuffer(
                        commandQueue,
                        inputImageBuffer,
                        CL_FALSE,
                        0,
                        width * height * pixelSize,
                        inputImageData,
                        0,
                        NULL,
                        &writeEvt);
            frameTaken = true;
            CHECK_OPENCL_ERROR(status, "clEnqueueWriteBuffer failed. (inputImageBuffer)");

            status = clFlush(commandQueue);
            CHECK_OPENCL_ERROR(status, "clFlush failed.");

            status = waitForEventAndRelease(&writeEvt);
            CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(writeEvt) Failed");
        
            // Set appropriate arguments to the kernel

            // input buffer image
            status = clSetKernelArg(
                        kernel,
                        0,
                        sizeof(cl_mem),
                        &inputImageBuffer);
            CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (inputImageBuffer)")

            // outBuffer imager
            status = clSetKernelArg(
                        kernel,
                        1,
                        sizeof(cl_mem),
                        &outputImageBuffer);
            CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (outputImageBuffer)");

            // float x = 0.5f;
            // status = clSetKernelArg(
            //             kernel,
            //             2,
            //             sizeof(float),
            //             &x);
            // CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (outputImageBuffer)");

            // Enqueue a kernel run call.
            size_t globalThreads[] = {width, height};
            size_t localThreads[] = {blockSizeX, blockSizeY};
        
            cl_event ndrEvt;
            status = clEnqueueNDRangeKernel(
                        commandQueue,
                        kernel,
                        2,
                        NULL,
                        globalThreads,
                        localThreads,
                        0,
                        NULL,
                        &ndrEvt);
            CHECK_OPENCL_ERROR(status, "clEnqueueNDRangeKernel failed.");

            status = clFlush(commandQueue);
            CHECK_OPENCL_ERROR(status, "clFlush failed.");

            status = waitForEventAndRelease(&ndrEvt);
            CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt) Failed");
        

            // Enqueue readBuffer
            cl_event readEvt;
            status = clEnqueueReadBuffer(
                        commandQueue,
                        outputImageBuffer,
                        CL_FALSE,
                        0,
                        width * height * pixelSize,
                        outputImageData,
                        0,
                        NULL,
                        &readEvt);
            CHECK_OPENCL_ERROR(status, "clEnqueueReadBuffer failed.");

            status = clFlush(commandQueue);
            CHECK_OPENCL_ERROR(status, "clFlush failed.");

            status = waitForEventAndRelease(&readEvt);
            CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(readEvt) Failed");
            
            outputImageToSave = outputImageData;
            outputImageReady = true;
        }
        cv::Mat mat = cv::Mat(height, width, CV_8UC4, outputImageData);
        
        cv::imshow("Sobel", mat);
        cv::waitKey(1);
        auto endTime = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime = endTime - startTime;
        kernelProcessingTimeAll += processingTime;
        no_kernel_calls++;
        // std::cout << "Processing time for frame: " << processingTime.count() << " seconds" << std::endl;
    }
    return SDK_SUCCESS;
}



int
SobelFilter::initialize()
{
    cl_int status = 0;
    // Call base class Initialize to get default configuration
    status = sampleArgs->initialize();
    CHECK_ERROR(status, SDK_SUCCESS, "OpenCL Initialization failed");

    Option* iteration_option = new Option;
    CHECK_ALLOCATION(iteration_option, "Memory Allocation error.\n");
    iteration_option->_sVersion = "i";
    iteration_option->_lVersion = "iterations";
    iteration_option->_description = "Number of iterations to execute kernel";
    iteration_option->_type = CA_ARG_INT;
    iteration_option->_value = &iterations;
    sampleArgs->AddOption(iteration_option);
    delete iteration_option;

    Option* kernel_file = new Option;
    CHECK_ALLOCATION(kernel_file, "Memory allocation error.\n");
    kernel_file->_sVersion = "k";
    kernel_file->_lVersion = "kerner_file";
    kernel_file->_description = "Kernel File (default: SobelFilter)";
    kernel_file->_type = CA_ARG_STRING;
    kernel_file->_value = &kernel_f;
    sampleArgs->AddOption(kernel_file);
    delete kernel_file;
    
    Option* camera_number = new Option;
    CHECK_ALLOCATION(camera_number, "Memory allocation error.\n");
    camera_number->_sVersion = "c";
    camera_number->_lVersion = "camera_num";
    camera_number->_description = "Index of camera (default: 0)";
    camera_number->_type = CA_ARG_INT;
    camera_number->_value = &camera_num;
    sampleArgs->AddOption(camera_number);
    delete camera_number;

    return SDK_SUCCESS;
}

int
SobelFilter::setup()
{
    cap_.open(camera_num);
    if (!cap_.isOpened())
    {
        throw std::invalid_argument( "Error: Couldn't open the camera" );
        return SDK_FAILURE;
    }

    camThread = std::thread([this]() { this->readInputImage(); });
    while (!camReady) {}
    cl_int status = 0;
    // Allocate host memory and read input image
    std::string filePath = getPath() + std::string(INPUT_IMAGE);
    
    // std::thread camThread([this]() { this->readInputImage(); });


    // status = readInputImage(filePath);
    // CHECK_ERROR(status, SDK_SUCCESS, "Read InputImage failed");

    // create and initialize timers
    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    status = setupCL();
    if(status != SDK_SUCCESS)
    {
        return status;
    }

    sampleTimer->stopTimer(timer);
    // Compute setup time
    setupTime = (double)(sampleTimer->readTimer(timer));

    return SDK_SUCCESS;
}


int
SobelFilter::run()
{
    cl_int status = 0;
    if(!byteRWSupport)
    {
        return SDK_SUCCESS;
    }
    while (!camReady) {}
    kernelThread = std::thread([this]() { this->runCLKernels(); });

    // runCLKernels();

    return SDK_SUCCESS;
}

int
SobelFilter::cleanup()
{
    if(!byteRWSupport)
    {
        return SDK_SUCCESS;
    }

    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;

    status = clReleaseKernel(kernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.");

    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.");

    status = clReleaseMemObject(inputImageBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.");

    status = clReleaseMemObject(outputImageBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.");

    status = clReleaseCommandQueue(commandQueue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.");

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "clReleaseContext failed.");

    // release program resources (input memory etc.)
    FREE(inputImageData);

    FREE(outputImageData);

    FREE(verificationOutput);

    FREE(devices);

    return SDK_SUCCESS;
}


void
SobelFilter::sobelFilterCPUReference()
{
    // x-axis gradient mask
    const int kx[][3] =
    {
        { 1, 2, 1},
        { 0, 0, 0},
        { -1,-2,-1}
    };

    // y-axis gradient mask
    const int ky[][3] =
    {
        { 1, 0, -1},
        { 2, 0, -2},
        { 1, 0, -1}
    };

    int gx = 0;
    int gy = 0;

    // pointer to input image data
    cl_uchar *ptr = (cl_uchar*)malloc(width * height * pixelSize);
    memcpy(ptr, inputImageData, width * height * pixelSize);

    // each pixel has 4 uchar components
    int w = width * 4;

    int k = 1;

    // apply filter on each pixel (except boundary pixels)
    for(int i = 0; i < (int)(w * (height - 1)) ; i++)
    {
        if(i < (k+1)*w - 4 && i >= 4 + k*w)
        {
            gx =  kx[0][0] **(ptr + i - 4 - w)
                  + kx[0][1] **(ptr + i - w)
                  + kx[0][2] **(ptr + i + 4 - w)
                  + kx[1][0] **(ptr + i - 4)
                  + kx[1][1] **(ptr + i)
                  + kx[1][2] **(ptr + i + 4)
                  + kx[2][0] **(ptr + i - 4 + w)
                  + kx[2][1] **(ptr + i + w)
                  + kx[2][2] **(ptr + i + 4 + w);


            gy =  ky[0][0] **(ptr + i - 4 - w)
                  + ky[0][1] **(ptr + i - w)
                  + ky[0][2] **(ptr + i + 4 - w)
                  + ky[1][0] **(ptr + i - 4)
                  + ky[1][1] **(ptr + i)
                  + ky[1][2] **(ptr + i + 4)
                  + ky[2][0] **(ptr + i - 4 + w)
                  + ky[2][1] **(ptr + i + w)
                  + ky[2][2] **(ptr + i + 4 + w);

            float gx2 = pow((float)gx, 2);
            float gy2 = pow((float)gy, 2);


            *(verificationOutput + i) = (cl_uchar)(sqrt(gx2 + gy2) / 2.0);
        }

        // if reached at the end of its row then incr k
        if(i == (k + 1) * w - 5)
        {
            k++;
        }
    }

    free(ptr);
}


int
SobelFilter::verifyResults()
{
    if(!byteRWSupport)
    {
        return SDK_SUCCESS;
    }

    if(sampleArgs->verify)
    {
        // reference implementation
        sobelFilterCPUReference();

        float *outputDevice = new float[width * height * pixelSize];
        CHECK_ALLOCATION(outputDevice,
                         "Failed to allocate host memory! (outputDevice)");

        float *outputReference = new float[width * height * pixelSize];
        CHECK_ALLOCATION(outputReference, "Failed to allocate host memory!"
                         "(outputReference)");

        // copy uchar data to float array
        for(int i = 0; i < (int)(width * height); i++)
        {
            outputDevice[i * 4 + 0] = outputImageData[i].s[0];
            outputDevice[i * 4 + 1] = outputImageData[i].s[1];
            outputDevice[i * 4 + 2] = outputImageData[i].s[2];
            outputDevice[i * 4 + 3] = outputImageData[i].s[3];

            outputReference[i * 4 + 0] = verificationOutput[i * 4 + 0];
            outputReference[i * 4 + 1] = verificationOutput[i * 4 + 1];
            outputReference[i * 4 + 2] = verificationOutput[i * 4 + 2];
            outputReference[i * 4 + 3] = verificationOutput[i * 4 + 3];
        }


        // compare the results and see if they match
        if(compare(outputReference,
                   outputDevice,
                   width * height * 4))
        {
            std::cout << "Passed!\n" << std::endl;
            delete[] outputDevice;
            delete[] outputReference;
            return SDK_SUCCESS;
        }
        else
        {
            std::cout << "Failed\n" << std::endl;
            delete[] outputDevice;
            delete[] outputReference;
            return SDK_FAILURE;
        }
    }

    return SDK_SUCCESS;
}

void SobelFilter::stop()
{
    stopCamThread = true;
    stopKernelThread = true;
    if (camThread.joinable())
    {
        camThread.join();
    }
    if (kernelThread.joinable())
    {
        kernelThread.join();
    }
}

void
SobelFilter::printStats()
{
    std::cout << std::endl;
    std::cout << "===========================================" << std::endl;
    std::cout << "Number of frames: " << no_frames << std::endl;
    std::cout << "Number of lost frames: " << lostFrames << std::endl;
    std::cout << "Number of kernel calls: " << no_kernel_calls << std::endl;
    std::cout << "Average time between frames: " << frameIntervalAll.count() / no_frames << " s" << std::endl;
    std::cout << "Average time of kernel: " << kernelProcessingTimeAll.count() / no_kernel_calls << " s" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    if(sampleArgs->timing)
    {
        std::string strArray[4] =
        {
            "Width",
            "Height",
            "Time(sec)",
            "[Transfer+Kernel]Time(sec)"
        };
        std::string stats[4];

        sampleTimer->totalTime = setupTime + kernelTime;

        stats[0] = toString(width, std::dec);
        stats[1] = toString(height, std::dec);
        stats[2] = toString(sampleTimer->totalTime, std::dec);
        stats[3] = toString(kernelTime, std::dec);

        printStatistics(strArray, stats, 4);
    }
}

cv::Mat SobelFilter::camera()
{
    cv::Mat frame;
    cap_ >> frame;

    if (frame.empty())
    {
        throw std::runtime_error("Error: Couldn't capture frame");
    }
    cv::imshow("Camera Feed", frame);
    return frame;
}

void SobelFilter::readKeys()
{
    while (!camReady) {}
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        key = cv::waitKey(1);
        if (key == 'q')
        {
            stop();
            cv::destroyAllWindows();
            break;
        }
        else if (key == 's')
        {
            writeInputImage("Input_image.bmp");
            writeOutputImage("Output_image.bmp");
            std::cout << "Saved image" << std::endl;
        }
        else if (key == 'd')
        {
            size_t i = 0;
            while (i < 10)
            {
                if (outputImageReady)
                {
                    std::ostringstream oss_input;
                    std::ostringstream oss_output;
                    oss_input << "Input_image_" << i << ".bmp";
                    oss_output << "Output_image_" << i << ".bmp";
                    std::string inputImageName = oss_input.str();
                    std::string outputImageName = oss_output.str();
                    writeInputImage(inputImageName);
                    writeOutputImage(outputImageName);
                    i++;
                }
            }
            std::cout << "Saved 10 images" << std::endl;
        }
    }
}

int main(int argc, char * argv[])
{
    cl_int status = 0;
    SobelFilter clSobelFilter;
    
    if(clSobelFilter.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clSobelFilter.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clSobelFilter.sampleArgs->isDumpBinaryEnabled())
    {
        return clSobelFilter.genBinaryImage();
    }
    
    status = clSobelFilter.setup();
    
    if(status != SDK_SUCCESS)
    {
        return status;
    }

    if(clSobelFilter.run() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    clSobelFilter.readKeys();
    
    if(clSobelFilter.verifyResults() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clSobelFilter.cleanup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }
    clSobelFilter.printStats();

    return SDK_SUCCESS;
}