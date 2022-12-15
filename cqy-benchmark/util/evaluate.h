#pragma once
#include <assert.h>
#include <chrono>
#include <numeric>
#include <math.h>
#include <sys/time.h>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))

#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif

namespace benchmark{
    // evaluate performance functions:

using namespace std::chrono;

class Timer {
public:
  typedef high_resolution_clock Clock;
   Timer() {start();}
  void start()
  { epoch = Clock::now(); }
  double time_elapsed_ms() const
  { 
    auto cost = Clock::now() - epoch; 
    return duration_cast<microseconds>(cost).count()/1000;
  }
  double time_elapsed_s() const
  { 
    auto cost = Clock::now() - epoch; 
    return duration_cast<seconds>(cost).count();
  }
private:
  Clock::time_point epoch;
};

    // get average and tp99 latancy
    inline std::pair<double, double>
    GetAverageAndTP99Latancy(double* latancy, uint32_t round_num) {
        auto average = std::accumulate(latancy, latancy + round_num, 0) / (double)round_num;
        std::sort(latancy, latancy + round_num);
        auto tp99_id = int(round_num * 0.99);
        tp99_id = (tp99_id == round_num ? round_num - 1 : tp99_id);
        return std::make_pair(average, latancy[tp99_id]);
    }
    
    /**
    * Returns the peak (maximum so far) resident set size (physical
    * memory use) measured in MB, or zero if the value cannot be
    * determined on this OS.
    */
    static size_t getPeakRSS() {
#if defined(_WIN32)
        /* Windows -------------------------------------------------- */
        PROCESS_MEMORY_COUNTERS info;
        GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
        return (size_t)info.PeakWorkingSetSize / (1024L * 1024L); 

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
        /* AIX and Solaris ------------------------------------------ */
        struct psinfo psinfo;
        int fd = -1;
        if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
            return (size_t)0L;      /* Can't open? */
        if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo))
        {
            close(fd);
            return (size_t)0L;      /* Can't read? */
        }
        close(fd);
        return (size_t)(psinfo.pr_rssize / 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
        /* BSD, Linux, and OSX -------------------------------------- */
        struct rusage rusage;
        getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
        return (size_t)rusage.ru_maxrss / (1024L * 1024L);
#else
        return (size_t) (rusage.ru_maxrss / 1024L);
#endif

#else
        /* Unknown OS ----------------------------------------------- */
        return (size_t)0L;          /* Unsupported. */
#endif
    }


    /**
    * Returns the current resident set size (physical memory use) measured
    * in MB, or zero if the value cannot be determined on this OS.
    */
    static size_t getCurrentRSS() {
#if defined(_WIN32)
        /* Windows -------------------------------------------------- */
        PROCESS_MEMORY_COUNTERS info;
        GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
        return (size_t)info.WorkingSetSize / (1024L * 1024L);

#elif defined(__APPLE__) && defined(__MACH__)
        /* OSX ------------------------------------------------------ */
        struct mach_task_basic_info info;
        mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
            (task_info_t)&info, &infoCount) != KERN_SUCCESS)
            return (size_t)0L;      /* Can't access? */
        return (size_t)info.resident_size / (1024L * 1024L);

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
        /* Linux ---------------------------------------------------- */
        long rss = 0L;
        FILE *fp = NULL;
        if ((fp = fopen("/proc/self/statm", "r")) == NULL)
            return (size_t) 0L;      /* Can't open? */
        if (fscanf(fp, "%*s%ld", &rss) != 1) {
            fclose(fp);
            return (size_t) 0L;      /* Can't read? */
        }
        fclose(fp);
        return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE) / (1024L * 1024L);

#else
        /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
        return (size_t)0L;          /* Unsupported. */
#endif
}
    // evaluate knn search functions:
    inline 
    float
    CalcKNNRecall(const int64_t* gt_ids, const int64_t* ids, int64_t nq, int64_t gt_k, int64_t k) {
        int64_t min_k = std::min(gt_k, k);
        int64_t hit = 0;
        for (auto i = 0; i < nq; i++) {
            std::unordered_set<int32_t> ground(gt_ids + i * gt_k, gt_ids + i * gt_k + min_k);
            for (auto j = 0; j < min_k; j++) {
                auto id = ids[i * k + j];
                if (ground.count(id) > 0) {
                    hit++;
                }
            }
        }
        return (hit * 1.0f / (nq * min_k));
    }

    inline
    float
    CalcKNNRecall(const int64_t* gt_ids, const int64_t* ids, int64_t nq_start, int64_t step, int64_t gt_k, int64_t k) {
        assert(nq_start + step <= 10000);
        int64_t min_k = std::min(gt_k, k);
        int64_t hit = 0;
        for (auto i = 0; i < step; i++) {
            std::unordered_set<int64_t> ground(gt_ids + (i + nq_start) * gt_k,
                                                gt_ids + (i + nq_start) * gt_k + min_k);
            for (auto j = 0; j < min_k; j++) {
                auto id = ids[i * k + j];
                if (ground.count(id) > 0) {
                    hit++;
                }
            }
        }
        return (hit * 1.0f / (step * min_k));
    }

    // evaluate range search functions:
    inline 
    int64_t
    CalcRangeSearchHits(const int64_t* gt_ids, const int64_t* ids, const size_t* gt_lims, const size_t* lims, const int64_t nq) {
        int64_t hit = 0;
        for (auto i = 0; i < nq; i++) {
            std::unordered_set<int64_t> gt_ids_set(gt_ids + gt_lims[i], gt_ids + gt_lims[i + 1]);
            for (auto j = lims[i]; j < lims[i + 1]; j++) {
                if (gt_ids_set.count(ids[j]) > 0) {
                    hit++;
                }
            }
        }
        return hit;
    }

    inline 
    int64_t
    CalcRangeSearchHits(const int64_t* gt_ids, const int64_t* ids, const size_t* gt_lims, const size_t* lims, int64_t start, int64_t num) {
        int64_t hit = 0;
        for (auto i = 0; i < num; i++) {
            std::unordered_set<int64_t> gt_ids_set(gt_ids + gt_lims[start + i], gt_ids + gt_lims[start + i + 1]);
            for (auto j = lims[i]; j < lims[i + 1]; j++) {
                if (gt_ids_set.count(ids[j]) > 0) {
                    hit++;
                }
            }
        }
        return hit;
    }
    
    inline 
    float
    CalcRangeSearchRecall(const int64_t* gt_ids, const int64_t* ids, const size_t* gt_lims, const size_t* lims, int64_t nq) {
        auto hit = CalcRangeSearchHits(gt_ids, ids, gt_lims, lims, nq);
        return (hit * 1.0f / gt_lims[nq]);
    }
    
    inline 
    float
    CalcRangeSearchAccuracy(const int64_t* gt_ids, const int64_t* ids, const size_t* gt_lims, const size_t* lims, int64_t nq) {
        auto hit = CalcRangeSearchHits(gt_ids, ids, gt_lims, lims, nq);
        return (hit * 1.0f / lims[nq]);
    }
};
