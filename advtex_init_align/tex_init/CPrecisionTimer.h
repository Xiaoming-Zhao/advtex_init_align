/*  This file is part of distributed Structured Prediction (dSP) - http://www.alexander-schwing.de/
 *
 *  distributed Structured Prediction (dSP) is free software: you can
 *  redistribute it and/or modify it under the terms of the GNU General
 *  Public License as published by the Free Software Foundation, either
 *  version 3 of the License, or (at your option) any later version.
 *
 *  distributed Structured Prediction (dSP) is distributed in the hope
 *  that it will be useful, but WITHOUT ANY WARRANTY; without even the
 *  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 *  PURPOSE. See the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with distributed Structured Prediction (dSP).
 *  If not, see <http://www.gnu.org/licenses/>.
 *
 *  Copyright (C) 2010-2013  Alexander G. Schwing  [http://www.alexander-schwing.de/]
 */

//Author: Alexander G. Schwing

#ifndef __CPRECISIONTIMER_H__
#define __CPRECISIONTIMER_H__

//#define NO_C11

#ifdef NO_C11
#ifdef _MSC_VER
#include <windows.h>
#else
#include <sys/time.h>
#endif
#else
#include <chrono>
#endif

class CPrecisionTimer
{
#ifdef NO_C11
#ifdef USE_ON_WINDOWS
    LARGE_INTEGER lFreq, lStart;
#else
    timeval lStart;
#endif
#else
    std::chrono::high_resolution_clock::time_point lStart;
#endif

public:
    CPrecisionTimer()
    {
#ifdef NO_C11
#ifdef USE_ON_WINDOWS
        QueryPerformanceFrequency(&lFreq);
#endif
#endif
    }

    inline void Start()
    {
#ifdef NO_C11
#ifdef USE_ON_WINDOWS
        QueryPerformanceCounter(&lStart);
#else
        gettimeofday(&lStart, 0);
#endif
#else
        lStart = std::chrono::high_resolution_clock::now();
#endif
    }

    inline double Stop()
    {
        // Return duration in seconds...
#ifdef NO_C11
#ifdef USE_ON_WINDOWS
        LARGE_INTEGER lEnd;
        QueryPerformanceCounter(&lEnd);
        return (double(lEnd.QuadPart - lStart.QuadPart) / lFreq.QuadPart);
#else
        timeval lFinish;
        gettimeofday(&lFinish, 0);
        return double(lFinish.tv_sec - lStart.tv_sec) + double(lFinish.tv_usec - lStart.tv_usec)/1e6; 
#endif
#else
        std::chrono::high_resolution_clock::time_point lEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double> >(lEnd - lStart);
        return time_span.count();
#endif
    }
};

#endif
