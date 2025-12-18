#include<immintrin.h>
#include "assert.h"

#include <chrono>
#include <cstddef>
#include <random>
#include <cstdlib>
#include <stdexcept>
#include <set>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <sys/stat.h>


namespace distance {

    static inline float __attribute__((always_inline))
    L2Sqr(const void *__restrict__ pVect1v,
                        const void *__restrict__ pVect2v,
                        size_t qty) {
        const float *p1 = static_cast<const float *>(pVect1v);
        const float *p2 = static_cast<const float *>(pVect2v);

        __m512 sum_vec0 = _mm512_setzero_ps();
        __m512 sum_vec1 = _mm512_setzero_ps();
        __m512 sum_vec2 = _mm512_setzero_ps();
        __m512 sum_vec3 = _mm512_setzero_ps();
        
        // Process 32 floats (2x 16-float vectors) per loop
        const size_t qty_unrolled = qty & ~63;
        const float *p1_end_unrolled = p1 + qty_unrolled;

        while (p1 < p1_end_unrolled) {
            // OPTIMIZATION: Calculate diff once and reuse
            __m512 diff0 = _mm512_sub_ps(_mm512_loadu_ps(p1), _mm512_loadu_ps(p2));
            sum_vec0 = _mm512_fmadd_ps(diff0, diff0, sum_vec0);
            
            __m512 diff1 = _mm512_sub_ps(_mm512_loadu_ps(p1 + 16), _mm512_loadu_ps(p2 + 16));
            sum_vec1 = _mm512_fmadd_ps(diff1, diff1, sum_vec1);

            __m512 diff2 = _mm512_sub_ps(_mm512_loadu_ps(p1 + 32), _mm512_loadu_ps(p2 + 32));
            sum_vec2 = _mm512_fmadd_ps(diff2, diff2, sum_vec2);
            
            __m512 diff3 = _mm512_sub_ps(_mm512_loadu_ps(p1 + 48), _mm512_loadu_ps(p2 + 48));
            sum_vec3 = _mm512_fmadd_ps(diff3, diff3, sum_vec3);
            
            p1 += 64;
            p2 += 64;
        }

        // Cleanup the remaining 16-float block, if any
        // const float *p1_end = p1 + (qty - qty_unrolled);
        // if (p1 < p1_end) {
        //     __m512 diff0 = _mm512_sub_ps(_mm512_loadu_ps(p1), _mm512_loadu_ps(p2));
        //     sum_vec0 = _mm512_fmadd_ps(diff0, diff0, sum_vec0);
        // }
        if (qty - qty_unrolled >= 32) {
            // OPTIMIZATION: Calculate diff once and reuse
            __m512 diff0 = _mm512_sub_ps(_mm512_loadu_ps(p1), _mm512_loadu_ps(p2));
            sum_vec0 = _mm512_fmadd_ps(diff0, diff0, sum_vec0);
            
            __m512 diff1 = _mm512_sub_ps(_mm512_loadu_ps(p1 + 16), _mm512_loadu_ps(p2 + 16));
            sum_vec1 = _mm512_fmadd_ps(diff1, diff1, sum_vec1);
            p1 += 32;
            p2 += 32;
        }
        if (qty - qty_unrolled >= 48 || (qty - qty_unrolled < 32 && qty - qty_unrolled >= 16)) {
            // OPTIMIZATION: Calculate diff once and reuse
            __m512 diff0 = _mm512_sub_ps(_mm512_loadu_ps(p1), _mm512_loadu_ps(p2));
            sum_vec0 = _mm512_fmadd_ps(diff0, diff0, sum_vec0);
            p1 += 16;
            p2 += 16;
        }
        size_t remainder = qty & 15; // 等价于 qty % 16
        if (remainder > 0) {
            // 创建一个掩码，其低`remainder`位为1，其余为0。
            // 例如，如果 remainder = 3, mask = 0b0111 = 7
            __mmask16 mask = (1 << remainder) - 1;

            // 使用掩码加载数据。只有掩码位为1的通道会被加载，其余通道的元素将被置零。
            // 这可以安全地处理边界，不会读取向量末端之外的内存。
            __m512 p1_rem = _mm512_maskz_loadu_ps(mask, p1);
            __m512 p2_rem = _mm512_maskz_loadu_ps(mask, p2);

            // 计算差值
            __m512 diff_rem = _mm512_sub_ps(p1_rem, p2_rem);

            // fmadd指令在这里不需要掩码，因为无效通道的差值已经是0，
            // 0*0 + sum = sum，不会影响最终结果。
            sum_vec0 = _mm512_fmadd_ps(diff_rem, diff_rem, sum_vec0);
        }

        sum_vec0 = _mm512_add_ps(sum_vec0, sum_vec1);
        sum_vec2 = _mm512_add_ps(sum_vec2, sum_vec3);
        sum_vec0 = _mm512_add_ps(sum_vec0, sum_vec2);


        return _mm512_reduce_add_ps(sum_vec0);
    }

}