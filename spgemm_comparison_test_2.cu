// spgemm_comparison_test.cu - Compare adj*feature_matrix SpGEMM
// Tests: Hash+AIA vs Hash-without-AIA vs cuSPARSE

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <string>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>

#include <cuda.h>
#include <helper_cuda.h>
#include <cusparse_v2.h>

#include <nsparse.hpp>
#include <CSR.hpp>
#include <SpGEMM.hpp>
#include <HashSpGEMM_volta.hpp>      // AIA implementation
#include <HashSpGEMM_volta_old.hpp>  // Non-AIA implementation

#include "graph_loader.hpp"

// Forward declarations for the functions we'll implement
template <class idType, class valType>
void SpGEMM_Hash_AIA_Timed(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c,
                           float &aia1_time, float &aia2_time);

template <class idType, class valType>                           
void SpGEMM_Hash_Old(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c);

typedef int IT;
#ifdef FLOAT
typedef float VT;
#else
typedef double VT;
#endif

// Performance measurement structure
struct PerformanceResult {
    float total_time_ms;
    float aia1_time_ms;
    float aia2_time_ms;
    float gflops;
    long long int flop_count;
    std::string implementation;
    std::string dataset;
    float sparsity;
    int feature_dim;
};

// Generate random sparse feature matrix with specified sparsity
CSR<IT, VT> generate_sparse_feature_matrix(int n_nodes, int feature_dim, float sparsity) {
    CSR<IT, VT> features;
    
    std::vector<IT> row_ptr(n_nodes + 1, 0);
    std::vector<IT> col_ids;
    std::vector<VT> values;
    
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<> col_dist(0, feature_dim - 1);
    std::uniform_real_distribution<VT> val_dist(-1.0, 1.0);
    std::uniform_real_distribution<float> sparse_dist(0.0, 1.0);
    
    for (int i = 0; i < n_nodes; i++) {
        for (int j = 0; j < feature_dim; j++) {
            if (sparse_dist(gen) > sparsity) {
                col_ids.push_back(j);
                values.push_back(val_dist(gen));
            }
        }
        row_ptr[i + 1] = col_ids.size();
    }
    
    // Initialize CSR structure
    features.nrow = n_nodes;
    features.ncolumn = feature_dim;
    features.nnz = col_ids.size();
    features.host_malloc = true;
    features.device_malloc = false;
    
    // Allocate and copy data
    features.rpt = new IT[n_nodes + 1];
    features.colids = new IT[features.nnz];
    features.values = new VT[features.nnz];
    
    std::copy(row_ptr.begin(), row_ptr.end(), features.rpt);
    std::copy(col_ids.begin(), col_ids.end(), features.colids);
    std::copy(values.begin(), values.end(), features.values);
    
    return features;
}

// Implementation of SpGEMM with AIA timing - modify volta.hpp functions
template <class idType, class valType>
void SpGEMM_Hash_AIA_Timed(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c,
                           float &aia1_time, float &aia2_time) {
    // This will call the volta.hpp version and extract AIA timings
    // For now, just call the regular SpGEMM_Hash and set timings to 0
    // You need to modify the volta.hpp to expose these timings
    SpGEMM_Hash(a, b, c);
    aia1_time = 0.0f;  // TODO: extract from modified volta.hpp
    aia2_time = 0.0f;  // TODO: extract from modified volta.hpp
}

// Implementation of SpGEMM without AIA - use volta_old.hpp
template <class idType, class valType>
void SpGEMM_Hash_Old(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c) {
    // This will call the volta_old.hpp version
    SpGEMM_Hash(a, b, c);  // This will use volta_old.hpp if that's included first
}

// Test function using Hash WITHOUT AIA (volta_old.hpp)
PerformanceResult test_spgemm_hash_old(CSR<IT, VT> adj, CSR<IT, VT> features, 
                                       const std::string& dataset, float sparsity) {
    PerformanceResult result;
    result.implementation = "Hash_without_AIA";
    result.dataset = dataset;
    result.sparsity = sparsity;
    result.feature_dim = features.ncolumn;
    
    IT i;
    long long int flop_count;
    cudaEvent_t event[2];
    float msec, ave_msec;
    
    for (i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }
    
    // Copy matrices to device
    adj.memcpyHtD();
    features.memcpyHtD();
    
    // Count FLOPs
    get_spgemm_flop(adj, features, flop_count);
    result.flop_count = flop_count;
    
    // Execute SpGEMM without AIA
    ave_msec = 0;
    CSR<IT, VT> output;
    
    for (i = 0; i < SpGEMM_TRI_NUM; i++) {
        if (i > 0) {
            output.release_csr();
        }
        
        cudaEventRecord(event[0], 0);
        // Use volta_old.hpp implementation directly
        #define USE_OLD_IMPLEMENTATION
        #include <HashSpGEMM_volta_old.hpp>
        SpGEMM_Hash(adj, features, output);
        #undef USE_OLD_IMPLEMENTATION
        cudaEventRecord(event[1], 0);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&msec, event[0], event[1]);
        
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= SpGEMM_TRI_NUM - 1;
    
    result.total_time_ms = ave_msec;
    result.aia1_time_ms = 0;  // No AIA
    result.aia2_time_ms = 0;
    result.gflops = (float)(flop_count) / 1000 / 1000 / ave_msec;
    
    output.memcpyDtH();
    output.release_csr();
    adj.release_csr();
    features.release_csr();
    
    for (i = 0; i < 2; i++) {
        cudaEventDestroy(event[i]);
    }
    
    return result;
}

// Test function using Hash WITH AIA (volta.hpp) with timing
PerformanceResult test_spgemm_hash_aia(CSR<IT, VT> adj, CSR<IT, VT> features, 
                                       const std::string& dataset, float sparsity) {
    PerformanceResult result;
    result.implementation = "Hash_with_AIA";
    result.dataset = dataset;
    result.sparsity = sparsity;
    result.feature_dim = features.ncolumn;
    
    IT i;
    long long int flop_count;
    cudaEvent_t event[2];
    float msec, ave_msec;
    float total_aia1 = 0, total_aia2 = 0;
    
    for (i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }
    
    // Copy matrices to device
    adj.memcpyHtD();
    features.memcpyHtD();
    
    // Count FLOPs
    get_spgemm_flop(adj, features, flop_count);
    result.flop_count = flop_count;
    
    // Execute SpGEMM with AIA and detailed timing
    ave_msec = 0;
    CSR<IT, VT> output;
    
    for (i = 0; i < SpGEMM_TRI_NUM; i++) {
        if (i > 0) {
            output.release_csr();
        }
        
        float aia1_time = 0, aia2_time = 0;
        
        cudaEventRecord(event[0], 0);
        SpGEMM_Hash_AIA_Timed(adj, features, output, aia1_time, aia2_time);  // Use AIA with timing
        cudaEventRecord(event[1], 0);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&msec, event[0], event[1]);
        
        if (i > 0) {
            ave_msec += msec;
            total_aia1 += aia1_time;
            total_aia2 += aia2_time;
        }
    }
    ave_msec /= SpGEMM_TRI_NUM - 1;
    
    result.total_time_ms = ave_msec;
    result.aia1_time_ms = total_aia1 / (SpGEMM_TRI_NUM - 1);
    result.aia2_time_ms = total_aia2 / (SpGEMM_TRI_NUM - 1);
    result.gflops = (float)(flop_count) / 1000 / 1000 / ave_msec;
    
    output.memcpyDtH();
    output.release_csr();
    adj.release_csr();
    features.release_csr();
    
    for (i = 0; i < 2; i++) {
        cudaEventDestroy(event[i]);
    }
    
    return result;
}

// Test function using cuSPARSE
PerformanceResult test_spgemm_cusparse(CSR<IT, VT> adj, CSR<IT, VT> features, 
                                       const std::string& dataset, float sparsity) {
    PerformanceResult result;
    result.implementation = "cuSPARSE";
    result.dataset = dataset;
    result.sparsity = sparsity;
    result.feature_dim = features.ncolumn;
    
    IT i;
    long long int flop_count;
    cudaEvent_t event[2];
    float msec, ave_msec;
    
    for (i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }
    
    // Copy matrices to device
    adj.memcpyHtD();
    features.memcpyHtD();
    
    // Count FLOPs
    get_spgemm_flop(adj, features, flop_count);
    result.flop_count = flop_count;
    
    // Execute cuSPARSE SpGEMM
    ave_msec = 0;
    CSR<IT, VT> output;
    
    for (i = 0; i < SpGEMM_TRI_NUM; i++) {
        if (i > 0) {
            output.release_csr();
        }
        
        cudaEventRecord(event[0], 0);
        SpGEMM_cuSPARSE(adj, features, output);
        cudaEventRecord(event[1], 0);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&msec, event[0], event[1]);
        
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= SpGEMM_TRI_NUM - 1;
    
    result.total_time_ms = ave_msec;
    result.aia1_time_ms = 0;  // No AIA
    result.aia2_time_ms = 0;
    result.gflops = (float)(flop_count) / 1000 / 1000 / ave_msec;
    
    output.memcpyDtH();
    output.release_csr();
    adj.release_csr();
    features.release_csr();
    
    for (i = 0; i < 2; i++) {
        cudaEventDestroy(event[i]);
    }
    
    return result;
}

// Results output functions
void print_results_header() {
    std::cout << std::setw(15) << "Dataset" 
              << std::setw(20) << "Implementation"
              << std::setw(10) << "Sparsity"
              << std::setw(8) << "FeatDim"
              << std::setw(12) << "Total(ms)"
              << std::setw(10) << "AIA1(ms)"
              << std::setw(10) << "AIA2(ms)"
              << std::setw(12) << "GFLOPS"
              << std::setw(12) << "Speedup"
              << std::endl;
    std::cout << std::string(120, '-') << std::endl;
}

void print_result(const PerformanceResult& result, float baseline_time = 0) {
    float speedup = (baseline_time > 0) ? baseline_time / result.total_time_ms : 1.0f;
    
    std::cout << std::setw(15) << result.dataset
              << std::setw(20) << result.implementation
              << std::setw(10) << std::fixed << std::setprecision(3) << result.sparsity
              << std::setw(8) << result.feature_dim
              << std::setw(12) << std::fixed << std::setprecision(2) << result.total_time_ms
              << std::setw(10) << std::fixed << std::setprecision(2) << result.aia1_time_ms
              << std::setw(10) << std::fixed << std::setprecision(2) << result.aia2_time_ms
              << std::setw(12) << std::fixed << std::setprecision(2) << result.gflops
              << std::setw(12) << std::fixed << std::setprecision(2) << speedup << "x"
              << std::endl;
}

// Main benchmark function
void run_spgemm_benchmark(const std::string& dataset_name, const std::string& data_path) {
    std::cout << "\n=== Testing Dataset: " << dataset_name << " ===" << std::endl;
    
    // Load adjacency matrix from spgemm-pruning format
    CSR<IT, VT> adj;
    if (dataset_name == "reddit") {
        adj = load_reddit_graph(data_path);
    } else if (dataset_name == "flickr") {
        adj = load_flickr_graph(data_path);
    } else if (dataset_name == "yelp") {
        adj = load_yelp_graph(data_path);
    } else if (dataset_name == "ogbn-products") {
        adj = load_ogbn_products_graph(data_path);
    } else if (dataset_name == "ogbn-proteins") {
        adj = load_ogbn_proteins_graph(data_path);
    } else {
        std::cerr << "Unknown dataset: " << dataset_name << std::endl;
        return;
    }
    
    std::cout << "Graph loaded: " << adj.nrow << " nodes, " << adj.nnz << " edges" << std::endl;
    
    // Test with different sparsities and feature dimension 256
    std::vector<float> sparsities = {0.5f, 0.25f, 0.125f, 0.0625f};
    int feature_dim = 256;
    
    for (float sparsity : sparsities) {
        std::cout << "\n--- Testing sparsity: " << sparsity << " ---" << std::endl;
        print_results_header();
        
        // Generate sparse feature matrix
        CSR<IT, VT> features = generate_sparse_feature_matrix(adj.nrow, feature_dim, sparsity);
        std::cout << "Feature matrix: " << features.nrow << "x" << features.ncolumn 
                  << ", nnz=" << features.nnz << " (sparsity=" << sparsity << ")" << std::endl;
        
        // Test all three implementations
        
        // 1. cuSPARSE baseline
        PerformanceResult cusparse_result = test_spgemm_cusparse(adj, features, dataset_name, sparsity);
        print_result(cusparse_result);
        
        // 2. Hash without AIA
        PerformanceResult hash_old_result = test_spgemm_hash_old(adj, features, dataset_name, sparsity);
        print_result(hash_old_result, cusparse_result.total_time_ms);
        
        // 3. Hash with AIA
        PerformanceResult hash_aia_result = test_spgemm_hash_aia(adj, features, dataset_name, sparsity);
        print_result(hash_aia_result, cusparse_result.total_time_ms);
        
        // Print AIA overhead analysis
        float aia_overhead = (hash_aia_result.aia1_time_ms + hash_aia_result.aia2_time_ms) / 
                            hash_aia_result.total_time_ms * 100.0f;
        std::cout << "AIA overhead: " << std::fixed << std::setprecision(1) << aia_overhead << "%" << std::endl;
        
        // Cleanup
        features.release_cpu_csr();
    }
    
    adj.release_cpu_csr();
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <data_path>" << std::endl;
        std::cout << "Example: " << argv[0] << " /path/to/spgemm-pruning/kernels/graphs" << std::endl;
        std::cout << "Data path should contain .indices and .indptr files for each dataset" << std::endl;
        return 1;
    }
    
    std::string data_path = argv[1];
    
    // Test datasets from spgemm-pruning
    std::vector<std::string> datasets = {"reddit", "flickr", "yelp", "ogbn-products", "ogbn-proteins"};
    
    std::cout << "SpGEMM Performance Comparison: adj * feature_matrix" << std::endl;
    std::cout << "Adjacency matrices from spgemm-pruning datasets" << std::endl;
    std::cout << "Feature matrix dimension: 256" << std::endl;
    std::cout << "Sparsity levels: 0.5, 0.25, 0.125, 0.0625" << std::endl;
    std::cout << "Comparing: Hash+AIA vs Hash-without-AIA vs cuSPARSE" << std::endl;
    
    for (const std::string& dataset : datasets) {
        try {
            run_spgemm_benchmark(dataset, data_path);
        } catch (const std::exception& e) {
            std::cerr << "Error testing " << dataset << ": " << e.what() << std::endl;
            continue;
        }
    }
    
    std::cout << "\nBenchmark completed!" << std::endl;
    return 0;
}
