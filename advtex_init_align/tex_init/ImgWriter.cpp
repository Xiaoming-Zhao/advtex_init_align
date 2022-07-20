#include <iostream>

#include "ImgWriter.h"

namespace NS_ImgWriter {

ImgWriter::ImgWriter(int nWorkers) : nWorkers(nWorkers), nActiveWorkers(0), keepRunning(true) {
    threadPool.resize(nWorkers);
    for(int k = 0; k < nWorkers; ++k) {
        threadPool[k] = new std::thread(&ImgWriter::WriteImg, this, k);
    }
}

ImgWriter::~ImgWriter() {
    // tell all threads to finish
    keepRunning = false;
    cvActive.notify_all();

    for(int k = 0; k < nWorkers; ++k) {
        threadPool[k]->join();
        delete threadPool[k];
        threadPool[k] = NULL;
    }
} 

void ImgWriter::AddImageToQueue(const boost::gil::rgb8_image_t& img, const std::string& fn, const ImgProcessTypeEnum& imgProcessType) {
    std::unique_lock<std::mutex> locker(mtx);
    dataQueue.push({new boost::gil::rgb8_image_t(img), new std::string(fn), new ImgProcessTypeEnum{imgProcessType}});
    // wake an idle thread
    cvActive.notify_one();
}

void ImgWriter::WriteImg(const int workerID) {

    WriterQueueData data(NULL, NULL, NULL);

    while (true) {
        std::unique_lock<std::mutex> locker(mtx);
        cvActive.wait(
            locker,
            [this](){return !keepRunning || !dataQueue.empty();}
        );

        if (!dataQueue.empty())
        {
            nActiveWorkers++;
            data = dataQueue.front();
            dataQueue.pop();
            locker.unlock();

            // async writing image
            std::cout << *data.fn << " [" << workerID << "]: " << data.img->width() << " x " << data.img->height() << std::endl;
            switch (*data.imgProcessType)
            {
                case ImgProcessTypeEnum::ROT90CCW:
                    boost::gil::write_view(*data.fn,
                        boost::gil::rotated90ccw_view(boost::gil::const_view(*data.img)),
                        boost::gil::png_tag());
                    break;
                case ImgProcessTypeEnum::TRANSPOSE_ROT90CCW:
                    boost::gil::write_view(*data.fn,
                        boost::gil::rotated90ccw_view(boost::gil::transposed_view(boost::gil::const_view(*data.img))),
                        boost::gil::png_tag());
                    break;
                default:
                    boost::gil::write_view(*data.fn, boost::gil::const_view(*data.img), boost::gil::png_tag());
            }

            delete data.imgProcessType;
            delete data.fn;
            delete data.img;

            locker.lock();
            nActiveWorkers--;
            locker.unlock();
            cvStop.notify_one();
        }
        else if (!keepRunning)
        {
            std::cout << "Kill thread " << workerID << std::endl;
            break;
        }
    }
}

void ImgWriter::WaitUntilFinish() {
    std::unique_lock<std::mutex> locker(mtx);
    cvStop.wait(
        locker,
        [this](){return dataQueue.empty() || nActiveWorkers == 0;}  // must put checking queue as the first flag
    );
}

} /* end namespace NS_ImgWriter */