#include "index.h"
#include <iostream>
#include <string> // Include the string header
#include "my_utils.h"
#include <vector>
#include <chrono>
#include <fstream>

std::vector<std::vector<int>> compute_groundtruth(const float* base_data, int num_base, const float* query_data, int num_query, int dim, int k) {
    std::vector<std::vector<int>> groundtruth(num_query, std::vector<int>(k));
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_query; ++i) {
        std::vector<std::pair<float, int>> distances;
        for (int j = 0; j < num_base; ++j) {
            float dist = distance::L2Sqr(query_data + i * dim, base_data + j * dim, dim);
            distances.emplace_back(dist, j);
        }
        // 按距离排序
        std::partial_sort(distances.begin(), distances.begin() + k, distances.end());
        // 取前 k 个最近邻
        for (int l = 0; l < k; ++l) {
            groundtruth[i][l] = distances[l].second;
        }
    }
    return groundtruth;
}

// 单线程计算 Groundtruth 的辅助函数
void compute_groundtruth_single_thread(const float* base_data, int num_base, const float* query_data, int start_idx, int end_idx, int dim, int k, std::vector<std::vector<int>>& groundtruth) {
    for (int i = start_idx; i < end_idx; ++i) {
        std::vector<std::pair<float, int>> distances;
        for (int j = 0; j < num_base; ++j) {
            float dist = distance::L2Sqr(query_data + i * dim, base_data + j * dim, dim);
            distances.emplace_back(dist, j);
        }
        // 使用 partial_sort 只排序前 k 个最近邻
        std::partial_sort(distances.begin(), distances.begin() + k, distances.end());
        for (int l = 0; l < k; ++l) {
            groundtruth[i][l] = distances[l].second;
        }
    }
}

// 多线程计算 Groundtruth 的函数
std::vector<std::vector<int>> compute_groundtruth_multithread(const float* base_data, int num_base, const float* query_data, int num_query, int dim, int k, int num_threads= std::thread::hardware_concurrency()) {
    std::vector<std::vector<int>> groundtruth(num_query, std::vector<int>(k));
    std::vector<std::thread> threads;
    int queries_per_thread = num_query / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        int start_idx = t * queries_per_thread;
        int end_idx = (t == num_threads - 1) ? num_query : start_idx + queries_per_thread;
        threads.emplace_back(compute_groundtruth_single_thread, base_data, num_base, query_data, start_idx, end_idx, dim, k, std::ref(groundtruth));
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return groundtruth;
}

float compute_recall(const std::vector<std::vector<int>>& groundtruth, const std::vector<std::vector<int>>& knn_results, int k) {
    int num_query = groundtruth.size();
    int correct_matches = 0;

    for (int i = 0; i < num_query; ++i) {
        std::unordered_set<int> groundtruth_set;
        for(auto && q : groundtruth[i]) {
            groundtruth_set.insert(q);
        }
        for (int j = 0; j < k; ++j) {
            if (groundtruth_set.find(knn_results[i][j]) != groundtruth_set.end()) {
                ++correct_matches;
            }
        }
    }
    return static_cast<float>(correct_matches) / (num_query * k);
}

// Function to save ground truth to a file
void save_gt(const std::string& filename, const std::vector<std::vector<int>>& gt) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Cannot open " << filename << " for writing" << std::endl;
        return;
    }
    size_t num_query = gt.size();
    size_t k = (num_query > 0) ? gt[0].size() : 0;
    out.write((char*)&num_query, sizeof(size_t));
    out.write((char*)&k, sizeof(size_t));
    for (const auto& vec : gt) {
        out.write((char*)vec.data(), k * sizeof(int));
    }
    out.close();
}

// Function to load ground truth from a file
bool load_gt(const std::string& filename, std::vector<std::vector<int>>& gt) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        return false;
    }
    size_t num_query, k;
    in.read((char*)&num_query, sizeof(size_t));
    in.read((char*)&k, sizeof(size_t));

    if (in.gcount() != sizeof(size_t)) return false; // Check if read was successful

    // Basic check for plausible dimensions to avoid excessive memory allocation
    if (num_query > 2000000 || k > 2000) {
         in.close();
         return false;
    }

    gt.assign(num_query, std::vector<int>(k));
    for (size_t i = 0; i < num_query; ++i) {
        in.read((char*)gt[i].data(), k * sizeof(int));
        if (in.gcount() != k * sizeof(int)) { // Check for read errors
            gt.clear();
            in.close();
            return false;
        }
    }
    in.close();
    return true;
}


int main(int argc,char* argv[]){
    std::cout<<"hello"<<std::endl;
    unsigned dim , num_data,num_query;
    int dim1;
    float * base = nullptr;
    // std::string data_path = "/home/zhouzhigan/dynamic_knn/dataset/sift/sift_base.fvecs";
    my_utils::load_data("/home/zhouzhigan/dynamic_knn/dataset/sift/sift_base.fvecs",base,num_data,dim);
    // num_data = 100000;
    // std::vector<float*> query;
    float * query = nullptr;
    // query = my_utils::read_fvecs("/home/zhouzhigan/dynamic_knn/dataset/sift/sift_query.fvecs",dim1);

    my_utils::load_data("/home/zhouzhigan/dynamic_knn/dataset/sift/sift_query.fvecs",query,num_query,dim);
    num_query = 1000;
    std::cout<<dim<<std::endl;

    Faster_HNSW::FHNSW * fhn = new Faster_HNSW::FHNSW(num_data,100,48,dim);
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) {
        auto time_start = std::chrono::high_resolution_clock::now();
        fhn->build(base,dim,num_data);
        auto time_end = std::chrono::high_resolution_clock::now();
        fhn->saveIndex(argv[1]);
        std::chrono::duration<double> elapsed = time_end - time_start;
        std::cout << "Build time: " << elapsed.count() << " seconds" << std::endl;
    }else{
        fhn->loadIndex(argv[1],base);
    }
    input.close();
    size_t count = 0 ; 
    size_t min = 100 ;
    size_t max = 0 ;
    // for(int i = 0 ; i < num_data; i++){
    //     count += *fhn->get_linkcount(i);
    //     if(*fhn->get_linkcount(i) < min)
    //         min = *fhn->get_linkcount(i);
    //     if(*fhn->get_linkcount(i) > max)
    //         max = *fhn->get_linkcount(i);
    // }
    std::cout<<"average link count: "<<(float)count/num_data<<std::endl;
    std::cout<<"min link count: "<<min<<std::endl;
    std::cout<<"max link count: "<<max<<std::endl;
    std::vector<Faster_HNSW::p_q_big> results(num_query);

    const char* gt_path = "gt.bin";
    int k = 10;
    std::vector<std::vector<int>> gt;

    if (!load_gt(gt_path, gt) || gt.size() != num_query || (gt.empty() ? 0 : gt[0].size()) != k) {
        if (!gt.empty()) {
            std::cout << "Ground truth parameters mismatch. Re-calculating..." << std::endl;
        } else {
            std::cout << "Ground truth file not found. Calculating..." << std::endl;
        }
        auto time_start_gt = std::chrono::high_resolution_clock::now();
        gt = compute_groundtruth(base, num_data, query, num_query, dim, k);
        auto time_end_gt = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_gt = time_end_gt - time_start_gt;
        std::cout << "Ground truth computation time: " << elapsed_gt.count() << " seconds" << std::endl;
        std::cout << "Saving ground truth to " << gt_path << std::endl;
        save_gt(gt_path, gt);
    } else {
        std::cout << "Loaded ground truth from " << gt_path << std::endl;
    }

    for(int ef = 10 ; ef < 20 ; ++ ef)
    {
        fhn->setEfSearch(ef);
        auto time_start_query = std::chrono::high_resolution_clock::now();
        for(int i = 0 ; i < num_query; i++){
            results[i] = fhn->searcknn(query + i * dim, 10);
        }
        auto time_end_query = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_query = time_end_query - time_start_query;
        std::cout << "Query time: " << elapsed_query.count() << " seconds" << std::endl;
        std::vector<std::vector<int>> knn_results(num_query);
        for(int i = 0 ; i < num_query; i++){
            while(!results[i].empty()){
                knn_results[i].push_back(static_cast<int>(results[i].top().second));
                results[i].pop();
            }
        }
        float recall = compute_recall(gt, knn_results, 10);
        std::cout << "Recall: " << recall << std::endl;
        std::cout << "QPS: " << num_query / elapsed_query.count() << std::endl;
    }
    return 0;
}