#ifndef H_IMG_WRITER2
#define H_IMG_WRITER2

#include <string>
#include <queue>
#include <thread> 
#include <mutex>
#include <atomic>
#include <condition_variable>

#include <boost/gil/image.hpp>
#include <boost/gil/typedefs.hpp>
#include <boost/gil/extension/io/png.hpp>

namespace NS_ImgWriter {

// We need to implement a producer-consumer concurrent mechanism.
// - producer: texture aligner that generates material images
// - consumer: threads to write image to disk

// Acknowledgements:
// - https://gist.github.com/ivcn/e793a7a4727e748fa65ff6870ece04f6
// - https://github.com/angrave/SystemProgramming/wiki/Synchronization%2C-Part-7%3A-The-Reader-Writer-Problem

// add processing operation on image from left to right
// e.g. TRANSPOSE_ROT90CCW will be boost::gil::rotated90ccw_view(boost::gil::transposed_view(...))
enum ImgProcessTypeEnum {NONE, ROT90CCW, TRANSPOSE_ROT90CCW};

class ImgWriter {
private:
    int nWorkers;
    unsigned int nActiveWorkers;

    std::mutex mtx;
    std::condition_variable cvActive;
    std::condition_variable cvStop;
    std::atomic<bool> keepRunning{true};
    
    struct WriterQueueData {
        const boost::gil::rgb8_image_t* img;
        const std::string* fn;
        const ImgProcessTypeEnum* imgProcessType;
        WriterQueueData(const boost::gil::rgb8_image_t* img, const std::string* fn, const ImgProcessTypeEnum* imgProcessType) : img(img), fn(fn), imgProcessType(imgProcessType) {}
    };

    std::queue<WriterQueueData> dataQueue;
    std::vector<std::thread*> threadPool;

    void WriteImg(const int wokerID);
public:
   ImgWriter(int nWorkers);
   ~ImgWriter();

   void AddImageToQueue(const boost::gil::rgb8_image_t& img, const std::string& fn, const ImgProcessTypeEnum& imgProcessType);

   void WaitUntilFinish();
};

} /* end namespace NS_ImgWriter */

#endif /* ifndef H_IMG_WRITER2 */