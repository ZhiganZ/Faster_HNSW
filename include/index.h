#include "distance.h"
#include <queue>

namespace Faster_HNSW{
    using tableint = unsigned int;
    using p_q_min = std::priority_queue<std::pair<float, tableint>, std::vector<std::pair<float, tableint>>, std::greater<std::pair<float, tableint>>>;
    using p_q_big = std::priority_queue<std::pair<float, tableint>, std::vector<std::pair<float, tableint>>, std::less<std::pair<float, tableint>>>;
    class FHNSW{
    private:
        // data layout [linksize,link,data,linksize,link,data,...]
        tableint ep;
        int max_level_;
        int max_elements_;
        int ef_construction_;
        int ef = 50;
        char * datalevel0{nullptr};     // 底层数据存储,max_elements_*(data_size+M*sizeof(tableint))
        int data_size;                  // 每个多维向量的大小, dim * sizeof(float)
        int size_per_element;           // 每个数据大小 data_size+M*sizeof(tableint)
        int M_;                         // 每个节点的最大连接数
        int dim_;
        int offest_data;                // 每个节点数据的偏移量, sizeof(int)+ M_*sizeof(tableint)
        char ** data_highlevel{nullptr};// 高层数据存储, data layout [linksize,link,data,tableint(下一层的标签),linksize,link,data,tableint(下一层的标签),...]

    public:
        FHNSW();
        FHNSW(int max_elements, int ef_construction, int M, int dim);
        virtual void init(float * data, int dim, int num_elements);
        virtual void build(float * data, int dim, int num_elements);
        virtual void searchKNN(float * query, int k);
        virtual p_q_big searchfor_neighbor(tableint id);
        virtual p_q_big searcknn(float * query, int k);
        virtual void saveIndex(const char * filename);
        virtual void loadIndex(const char * filename,const float * data);
        virtual void setEfSearch(int ef_search);
        virtual void iter();
        virtual void calculate_data();
        virtual void get_neighbor_using_vamana(p_q_big &top_candidates, size_t M);
        virtual inline float * get_data(tableint id);
        virtual inline void set_linkcount(tableint id, int linkcount);
        virtual inline tableint * get_linkcount(tableint id);
        virtual ~FHNSW();
    };
}