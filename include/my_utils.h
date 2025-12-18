#include <iostream>
#include <vector>

namespace my_utils {
    std::vector<float*> read_fvecs(const std::string& filename, int& dimension) {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        std::vector<float*> data;
        while (ifs.peek() != EOF) {
            int32_t d;
            ifs.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
            if (ifs.eof()) break; // Handle possible EOF after reading d

            dimension = d; // Assuming all vectors have the same dimension
            float* vec = (float*)malloc(d * sizeof(float));
            ifs.read(reinterpret_cast<char*>(vec), sizeof(float) * d);
            if (ifs.eof()) {
                free(vec); // Free allocated memory if EOF reached unexpectedly
                throw std::runtime_error("Unexpected end of file while reading vector data.");
            }
            data.emplace_back(vec);
        }

        ifs.close();
        return data;
    }
    void load_data(char* filename, float*& data, unsigned& num,
                unsigned& dim) {  // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char*)&dim, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);
    data = new float[(size_t)num * (size_t)dim];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(data + i * dim), dim * 4);
    }
    in.close();
    }
}