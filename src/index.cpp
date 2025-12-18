#include "index.h"

#include <omp.h>
#include <bitset>
#include <chrono>
#include <cmath>
#include <boost/dynamic_bitset.hpp>
#include <random>
#include <stdexcept> // Include this header for std::runtime_error
#include <fstream>
#include <string>
namespace Faster_HNSW{
    int generate_random_int(int min, int max) {
        // 1. 使用静态变量，确保只初始化（播种）一次
        //    这比每次调用都创建新引擎要高效得多
        static std::random_device rd;  // 获取一个高质量的随机数种子
        static std::mt19937 gen(rd()); // 使用 rd 来播种 Mersenne Twister 引擎

        // 2. 创建一个均匀分布的对象
        std::uniform_int_distribution<> distrib(min, max);

        // 3. 使用引擎和分布对象生成随机数
        return distrib(gen);
    }
    FHNSW::FHNSW(int max_elements, int ef_construction,int M,int dim):M_(M),ef_construction_(ef_construction),max_elements_(max_elements),dim_(dim) {}

    FHNSW::FHNSW() {}

    FHNSW::~FHNSW() {
        if(datalevel0!=nullptr)free(datalevel0);
    }

    void FHNSW::calculate_data(){
        int min = std::numeric_limits<int>::max();
        int max = 0;
        int count = 0;
        for(size_t i = 0; i < max_elements_; i++) {
            if(i % 10000 == 0) {
                std::cout << "FHNSW::calculate_data: Processing element " << i << " of " << max_elements_ << "." << std::endl;
            }
            tableint * link = get_linkcount(i);
            int size = *link;
            if(size < min) {
                min = size;
            }
            if(size > max) {
                max = size;
            }
            count += size;
        }
        std::cout << "FHNSW::calculate_data: Minimum link size: " << min << ", Maximum link size: " << max << ", Avg links: " << (double)count/max_elements_ << "." << std::endl;
    }

    struct neighbor{
        float dist;
        tableint id;
        bool operator < (const neighbor &other) const {
            return dist < other.dist;
        }
        bool operator == (const neighbor &other) const {
            return id == other.id;
        }
        bool operator == (const tableint &other_id) const {
            return id == other_id;
        }
        bool operator != (const neighbor &other) const {
            return id != other.id;
        }
    };

    void FHNSW::iter() {
        if(datalevel0 == nullptr) {
            throw std::runtime_error("datalevel0 is not initialized");
        }
        std::cout<<"FHNSW::iter: Starting iteration over " << max_elements_ << " elements." << std::endl;
        std::vector<std::vector<neighbor>> new_link(max_elements_);
        // std::vector<p_q_big> all_top_candidates(max_elements_);
        int min=1000;
        int max=0;
        #pragma omp parallel for schedule(dynamic)
        for(size_t i = 0; i < max_elements_; i++) {
            if(i% 10000 == 0) {
                std::cout<<"FHNSW::iter: Processing element " << i << " of " << max_elements_ << "." << std::endl;
            }
            p_q_big top = searchfor_neighbor(i);
            if(top.size()<ef_construction_/2){
                throw std::runtime_error("FHNSW::iter: Not enough neighbors found for element " + std::to_string(i) + ". Found: " + std::to_string(top.size()) + ", expected at least " + std::to_string(ef_construction_/2));
            }
            get_neighbor_using_vamana(top, M_);
            p_q_big tmp = std::move(top);
            // all_top_candidates[i] = std::move(top);
            // int count = 0 ;
            while(tmp.size() > 0) {
                if(tmp.top().second >= max_elements_ || tmp.top().second < 0) {
                    throw std::out_of_range("Link index out of range: " + std::to_string(tmp.top().second) + " for element " + std::to_string(i));
                }
                new_link[i].emplace_back(neighbor{tmp.top().first, tmp.top().second});
                tmp.pop();
            }
        }
        calculate_data();
        // for(auto && top : new_link) {
        //     if(top.size() < min) {
        //         min = top.size();
        //     }
        //     if(top.size() > max) {
        //         max = top.size();
        //     }
        // }


        // std::cout<<"FHNSW::iter: Minimum link size: " << min << ", Maximum link size: " << max << "." << std::endl;
        // sleep(5);
        std::vector<std::mutex> link_mutex(max_elements_);
        std::vector<std::vector<neighbor>> back_links(max_elements_);
        std::vector<std::mutex> back_link_mutex(max_elements_); // 为临时结构也准备一套锁

        // 阶段一：并行收集反向链接，只读 new_link，只写 back_links
        #pragma omp parallel for schedule(dynamic)
        for (tableint i = 0; i < max_elements_; i++) {
            if (i % 10000 == 0) {
                std::cout << "双向连接 [阶段1/3 - 收集]: " << i << " of " << max_elements_ << "." << std::endl;
            }

            // 在这个循环中，new_link 是只读的，所以是线程安全的。
            for (const auto& nei : new_link[i]) {
                tableint neighbor_id = nei.id;

                // 检查ID是否有效，这是一个很好的健壮性措施
                if (neighbor_id < 0 || neighbor_id >= max_elements_) {
                    // 如果在这里发现无效ID，说明问题出在更早的步骤
                    // (比如 searchfor_neighbor 或 get_neighbor_using_vamana)
                    // 但根据你的问题，问题主要出在链接阶段
                    continue; 
                }

                // 我们要为 neighbor_id 添加一个指向 i 的反向链接
                // 这需要修改 back_links[neighbor_id]，所以我们锁住它
                std::lock_guard<std::mutex> lock(back_link_mutex[neighbor_id]);
                back_links[neighbor_id].emplace_back(neighbor{nei.dist, i});
            }
        }

        // OMP `parallel for` 结束时有一个隐式的屏障，确保所有线程都完成了上面的操作

        // 阶段二：并行合并反向链接到主链接列表
        #pragma omp parallel for schedule(dynamic)
        for (tableint i = 0; i < max_elements_; i++) {
            if (i % 10000 == 0) {
                std::cout << "双向连接 [阶段2/3 - 合并]: " << i << " of " << max_elements_ << "." << std::endl;
            }
            // 每个线程只修改自己的 new_link[i]，不存在竞争
            new_link[i].insert(new_link[i].end(), back_links[i].begin(), back_links[i].end());
        }

        // 阶段三：去重和剪枝 (这部分在你的原始代码中已经有了)
        // 合并后，一个节点的邻居列表可能会超过 M_，并且可能包含重复项。
        // 你接下来的剪枝步骤会处理这个问题。
        #pragma omp parallel for schedule(dynamic)
        for(size_t i = 0; i < max_elements_; i++) {
            if (i % 10000 == 0) {
                std::cout << "双向连接 [阶段3/3 - 剪枝]: " << i << " of " << max_elements_ << "." << std::endl;
            }
            p_q_big top;
            // 使用一个 set 来自动去重
            std::set<tableint> visited_ids;
            for(auto && nei : new_link[i]) {
                if(visited_ids.find(nei.id) == visited_ids.end()) {
                    top.emplace(nei.dist, nei.id);
                    visited_ids.insert(nei.id);
                }
            }
            
            get_neighbor_using_vamana(top, M_); // 剪枝到 M_ 个邻居
            
            new_link[i].clear();
            while(!top.empty()) {
                new_link[i].emplace_back(neighbor{top.top().first, top.top().second});
                top.pop();
            }
            // 反转一下，让距离近的在前面，虽然对于后续逻辑不是必须的，但更符合直觉
            std::reverse(new_link[i].begin(), new_link[i].end());
        }
        // #pragma omp parallel for schedule(dynamic)
        // for (tableint i = 0; i < max_elements_; i++) {
        //     if (i % 10000 == 0) {
        //         std::cout << "双向连接 " << i << " of " << max_elements_ << "." << std::endl;
        //     }

        //     // for (size_t j = 0; j < new_link[i].size(); j++) {
        //     for(auto nei : new_link[i]) {
        //         // tableint neighbor = new_link[i][j];
        //         tableint cur_neighbor = nei.id;

        //         // 检查 neighbor 是否已经包含 i，避免重复连接
        //         bool exists = false;
        //         {
        //             std::lock_guard<std::mutex> lock(link_mutex[cur_neighbor]);
        //             exists = std::find(new_link[cur_neighbor].begin(), new_link[cur_neighbor].end(), i) != new_link[cur_neighbor].end();
        //         }
        //         if (exists) continue;

        //         // 添加双向连接
        //         {
        //             std::lock_guard<std::mutex> lock(link_mutex[cur_neighbor]);
        //             new_link[cur_neighbor].emplace_back(neighbor{nei.dist, i});
        //         }

        //         // 计算距离并更新候选集
        //         // float dist = distance::L2Sqr(get_data(cur_neighbor), get_data(i), dim_);
        //         // {
        //         //     std::lock_guard<std::mutex> lock(link_mutex[cur_neighbor]);
        //         //     all_top_candidates[cur_neighbor].emplace(dist, i);
        //         // }
        //     }
        // }
        #pragma omp parallel for schedule(dynamic)
        for(size_t i = 0; i < max_elements_; i++) {
            p_q_big top ;
            for(auto && nei : new_link[i]) {
                top.emplace(nei.dist, nei.id);
            }
            get_neighbor_using_vamana(top, M_);
            new_link[i].clear();
            int count = 0 ;
            while(top.size() > 0) {
                new_link[i].emplace_back(neighbor{top.top().first, top.top().second});
                top.pop();
            }
        }
        
        #pragma omp parallel for schedule(dynamic)
        for(size_t i = 0; i < max_elements_; i++) {
            set_linkcount(i, new_link[i].size());
            tableint * link = (tableint*)get_linkcount(i) + 1;
            // memcpy(link, new_link[i].data(), new_link[i].size() * sizeof(tableint));
            int count = 0 ;
            for(auto && neighbor : new_link[i]) {
                link[count++] = neighbor.id;
            }
        }
    }

    void FHNSW::build(float* data, int dim, int num_elements) {
        data_size = dim * sizeof(float);
        size_per_element = data_size + sizeof(tableint) * M_ + sizeof(tableint);
        offest_data = sizeof(tableint) * M_ + sizeof(tableint);
        max_elements_ = num_elements;
        datalevel0 = (char*)malloc(num_elements * size_per_element);
        if(datalevel0 == nullptr) {
            throw std::runtime_error("Memory allocation failed for datalevel0");
        }
        memset(datalevel0, 0, num_elements * size_per_element);
        std::cout<<"FHNSW::build: Allocated memory for " << num_elements << " elements, each of size " << size_per_element << " bytes." << std::endl;
        // #pragma omp parallel for schedule(dynamic)
        for(size_t i = 0; i < num_elements; i++) {
            memcpy(datalevel0 + i * size_per_element + offest_data, data + i * dim, data_size);
        }
        std::cout<<"FHNSW::build: Copied data to datalevel0." << std::endl;
        init(data, dim, num_elements);
        calculate_data();
        iter();
        calculate_data();
        iter();
        iter();
    }

    void FHNSW::init(float * data, int dim, int num_elements) {
        float * centroid = (float*)malloc(dim * sizeof(float));
        std::cout<<"FHNSW::init: Allocated memory for centroid of size " << dim * sizeof(float) << " bytes." << std::endl;
        for(size_t i = 0 ; i < num_elements; i++) {
            // #pragma omp parallel for schedule(dynamic)
            for(size_t j = 0; j < dim; j++) {
                centroid[j] += data[i * dim + j];
            }
        }
        for(size_t j = 0; j < dim; j++) {
            centroid[j] /= num_elements;
        }
        tableint new_ep = 0;
        std::cout<<"FHNSW::init: Calculating initial entry point." << std::endl;
        float min_size = distance::L2Sqr(centroid, data, dim);
        for(size_t i = 1 ; i < num_elements; i++) {
            float dist = distance::L2Sqr(centroid, data + i * dim, dim);
            if(dist < min_size) {
                min_size = dist;
                new_ep = i;
            }
        }
        ep = new_ep;
        std::cout<<"FHNSW::init: Initial entry point set to " << ep << " with distance " << min_size << "." << std::endl;
        #pragma omp parallel for schedule(dynamic)
        for(size_t i = 0 ; i < num_elements; i++) {
            if(i% 10000 == 0) {
                std::cout<<"FHNSW::init: Processing element " << i << " of " << num_elements << "." << std::endl;
            }
            std::set<tableint> links;
            p_q_big top_candidates;
            while(links.size()<M_){
                tableint random_link = generate_random_int(0, num_elements - 1);
                if(random_link == i ) continue;
                links.insert(random_link);
            }
            // for(auto && link : links) {
            //     top_candidates.emplace(distance::L2Sqr(get_data(i), get_data(link), dim),link);
            // }
            // get_neighbor_using_vamana(top_candidates, M_);
            set_linkcount(i, links.size());
            tableint * link = (tableint*)(datalevel0 + i * size_per_element + sizeof(tableint));
            // for(int i = 0 ; i < top_candidates.size(); i++) {
            //     link[i] = top_candidates.top().second;
            //     top_candidates.pop();
            // }
            int count = 0;
            for(auto && link_ : links) {
                if(link_ == i) continue; // 避免自连接
                link[count++] = link_;
            }
        }
    }

    p_q_big FHNSW::searchfor_neighbor(tableint id){
        boost::dynamic_bitset<> flags{max_elements_, 0};
        p_q_big top_candidates;
        p_q_min CandidateSet;

        _mm_prefetch((const char*)get_data(id), _MM_HINT_T0);
        _mm_prefetch((const char*)get_data(ep), _MM_HINT_T0);

        float lowerbound = distance::L2Sqr(get_data(id), get_data(ep), dim_);
        if(id!=ep)
            top_candidates.emplace(lowerbound, ep);
        else{
            lowerbound = std::numeric_limits<float>::max();
        }
        CandidateSet.emplace(lowerbound, ep);
        flags[ep] = 1;
        while(!CandidateSet.empty()) {
            auto current = CandidateSet.top();
            if(current.first > lowerbound&&top_candidates.size() == ef_construction_) {
                break;
            }
            CandidateSet.pop();

            if(current.second == id && id!=ep) continue;

            tableint * link = get_linkcount(current.second);

            int size = *link;
            // if(size!=M_){
            //     throw std::runtime_error("FHNSW::searchfor_neighbor: Link size is less than M_. This should not happen.");
            // }

            _mm_prefetch((const char*)get_data(link[1]), _MM_HINT_T0);

            for(int i = 1; i <=size ; i++) {
                tableint neighbor = link[i];
                _mm_prefetch((const char*)get_data(link[i+1]), _MM_HINT_T0);
                if(flags[neighbor]) continue;
                flags[neighbor] = 1;
                float dist = distance::L2Sqr(get_data(id), get_data(neighbor), dim_);
                CandidateSet.emplace(dist, neighbor);
                if(dist<lowerbound || top_candidates.size() < ef_construction_) {
                    top_candidates.emplace(dist, neighbor);
                
                    if (top_candidates.size() > ef_construction_){
                        top_candidates.pop();
                    }

                    lowerbound = top_candidates.top().first;
                }
            }
        }
        return top_candidates;
    }

    p_q_big FHNSW::searcknn(float* query, int k){
        boost::dynamic_bitset<> flags{max_elements_, 0};
        p_q_big top_candidates;
        p_q_min CandidateSet;

        // _mm_prefetch((const char*)get_data(ep), _MM_HINT_T0);

        float lowerbound = distance::L2Sqr(query, get_data(ep), dim_);
        top_candidates.emplace(lowerbound, ep);
        CandidateSet.emplace(lowerbound, ep);
        flags[ep] = 1;
        while(!CandidateSet.empty()) {
            auto current = CandidateSet.top();
            if(current.first > lowerbound&&top_candidates.size() == ef) {
                break;
            }
            CandidateSet.pop();

            tableint * link = get_linkcount(current.second);

            int size = *link;
            // if(size!=M_){
            //     throw std::runtime_error("FHNSW::searchfor_neighbor: Link size is less than M_. This should not happen.");
            // }

            // _mm_prefetch((const char*)get_data(link[1]), _MM_HINT_T0);

            for(int i = 1; i <=size ; i++) {
                tableint neighbor = link[i];
                // _mm_prefetch((const char*)get_data(link[i+1]), _MM_HINT_T0);
                if(flags[neighbor]) continue;
                flags[neighbor] = 1;
                float dist = distance::L2Sqr(query, get_data(neighbor), dim_);
                CandidateSet.emplace(dist, neighbor);
                if(dist<lowerbound || top_candidates.size() < ef) {
                    top_candidates.emplace(dist, neighbor);
                
                    if (top_candidates.size() > ef){
                        top_candidates.pop();
                    }

                    lowerbound = top_candidates.top().first;
                }
            }
        }
        while(top_candidates.size() > k) {
            top_candidates.pop();
        }
        return top_candidates;
    }

    void FHNSW::get_neighbor_using_vamana(p_q_big &top_candidates, size_t M){
        if (top_candidates.size() <= M) {
            return;
        }

        // 1. 数据结构转换：从优先队列转为有序的 vector
        // 这是为了能方便地使用下标访问和更新遮挡因子
        std::vector<std::pair<float, tableint>> candidate_pool;
        candidate_pool.reserve(top_candidates.size());
        while (!top_candidates.empty()) {
            candidate_pool.push_back(top_candidates.top());
            top_candidates.pop();
        }
        // `top_candidates` 是最大堆（距离大的在顶），所以需要反转得到从小到大的顺序
        std::reverse(candidate_pool.begin(), candidate_pool.end());


        // 2. 初始化遮挡因子数组和返回列表
        std::vector<std::pair<float, tableint>> return_list;
        return_list.reserve(M);
        
        // occlude_factor 与 candidate_pool 中的每个候选项一一对应
        std::vector<float> occlusion_factors(candidate_pool.size(), 0.0f);


        // 3. 迭代剪枝主循环 (Vamana/DiskANN的核心)
        float cur_alpha = 1.0f;
        while (cur_alpha <= 1.2 && return_list.size() < M) {
            
            for (size_t i = 0; i < candidate_pool.size() && return_list.size() < M; ++i) {
                
                // 如果当前候选项的遮挡因子已经超过了本轮标准，跳过
                if (occlusion_factors[i] > cur_alpha) {
                    continue;
                }

                // 标记为已选择（设为极大值），并加入返回列表
                occlusion_factors[i] = std::numeric_limits<float>::max();
                const auto& curent_pair = candidate_pool[i];
                return_list.push_back(curent_pair);

                // 4. 核心遮挡逻辑：用新选中的点去更新所有后续候选点的遮挡因子
                for (size_t j = i + 1; j < candidate_pool.size(); ++j) {
                    // 如果后续这个点已经被选了，也跳过
                    if (occlusion_factors[j] > 1.2) {
                        continue;
                    }

                    const auto& other_pair = candidate_pool[j];
                    
                    // 计算新选中的点 curent_pair 与另一个候选点 other_pair 之间的距离
                    float dist_between_candidates =
                            distance::L2Sqr(get_data(curent_pair.second),
                                        get_data(other_pair.second),
                                        dim_);
                    
                    // other_pair.first 是它到查询点的距离
                    float dist_to_query_of_other = other_pair.first;
                    
                    // 避免除以零
                    if (dist_between_candidates < 1e-6) continue;

                    // 计算并更新遮挡率: dist(query, other) / dist(current, other)
                    float current_occlusion_ratio = dist_to_query_of_other / dist_between_candidates;
                    
                    if (current_occlusion_ratio > occlusion_factors[j]) {
                        occlusion_factors[j] = current_occlusion_ratio;
                    }
                }
            }
            // 如果邻居还不够，增加 alpha 值，进入下一轮更严格的筛选
            cur_alpha *= 1.2f;
        }

        // 5. 将最终剪枝后的列表重新放回 top_candidates
        // 注意：原始的 top_candidates 在第1步已经被清空了
        for (const auto& pair : return_list) {
            top_candidates.push(pair);
        }
    }

    inline float * FHNSW::get_data(tableint id){
        return (float*)(datalevel0 + id * size_per_element + offest_data);
    }

    inline void FHNSW::set_linkcount(tableint id, int linkcount){
        // if(id < 0 || id >= max_elements_) {
        //     throw std::out_of_range("ID is out of range");
        // }
        tableint* link_count_ptr = (tableint*)(datalevel0 + id * size_per_element);
        *link_count_ptr = linkcount;
    }

    inline tableint * FHNSW::get_linkcount(tableint id){
        return (tableint*)(datalevel0 + id * size_per_element);
    }

    void FHNSW::searchKNN(float* query, int k) {
        // searchKNN implementation
    }

    void FHNSW::saveIndex(const char* filename) {
        std::ofstream output(filename, std::ios::binary);
        if (!output) {
            throw std::runtime_error(std::string("Cannot open file: ") + filename);
        }

        // Save metadata
        output.write((char*)&ep, sizeof(ep));
        output.write((char*)&max_level_, sizeof(max_level_));
        output.write((char*)&max_elements_, sizeof(max_elements_));
        output.write((char*)&ef_construction_, sizeof(ef_construction_));
        output.write((char*)&ef, sizeof(ef));
        output.write((char*)&data_size, sizeof(data_size));
        output.write((char*)&size_per_element, sizeof(size_per_element));
        output.write((char*)&M_, sizeof(M_));
        output.write((char*)&dim_, sizeof(dim_));
        output.write((char*)&offest_data, sizeof(offest_data));

        // Save only the link data for each element
        for (int i = 0; i < max_elements_; ++i) {
            output.write(datalevel0 + (size_t)i * size_per_element, offest_data);
        }

        output.close();
    }

    void FHNSW::loadIndex(const char* filename,const float * data) {
        std::ifstream input(filename, std::ios::binary);
        if (!input) {
            throw std::runtime_error(std::string("Cannot open file: ") + filename);
        }

        // Load metadata
        input.read((char*)&ep, sizeof(ep));
        input.read((char*)&max_level_, sizeof(max_level_));
        input.read((char*)&max_elements_, sizeof(max_elements_));
        input.read((char*)&ef_construction_, sizeof(ef_construction_));
        input.read((char*)&ef, sizeof(ef));
        input.read((char*)&data_size, sizeof(data_size));
        input.read((char*)&size_per_element, sizeof(size_per_element));
        input.read((char*)&M_, sizeof(M_));
        input.read((char*)&dim_, sizeof(dim_));
        input.read((char*)&offest_data, sizeof(offest_data));

        // Allocate memory for the graph
        if (datalevel0 != nullptr) {
            free(datalevel0);
        }
        datalevel0 = (char*)malloc((size_t)max_elements_ * size_per_element);
        if (datalevel0 == nullptr) {
            throw std::runtime_error("Memory allocation failed for datalevel0 in loadIndex");
        }

        if (data == nullptr) {
            throw std::runtime_error("Input data cannot be null for loadIndex");
        }

        // Load link data and copy vector data
        for (int i = 0; i < max_elements_; ++i) {
            // Read links
            input.read(datalevel0 + (size_t)i * size_per_element, offest_data);
            // Copy vectors
            memcpy(datalevel0 + (size_t)i * size_per_element + offest_data, data + (size_t)i * dim_, data_size);
        }

        input.close();
    }

    void FHNSW::setEfSearch(int ef_search) {
        ef = ef_search;
    }
}
