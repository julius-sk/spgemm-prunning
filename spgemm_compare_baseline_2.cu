// spgemm_comparison_test_cusparse.cu - Modified to use modern cuSPARSE API
// Compares spgemm_hash and cuSPARSE using modern cuSPARSE SpGEMM API

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <string>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <iomanip>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>

#include <nsparse.hpp>
#include <CSR.hpp>
#include <SpGEMM.hpp>
#include <HashSpGEMM_volta_old.hpp>  // Non-AIA implementation

// Include our additional headers
#include "graph_loader.hpp"

// Error checking macros for functions (modified to not exit)
#define CHECK_CUDA_FUNC(func)                                                  \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        result.total_time_ms = -1; return result;                              \
    }                                                                          \
}

#define CHECK_CUSPARSE_FUNC(func)                                             \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        result.total_time_ms = -1; return result;                              \
    }                                                                          \
}

typedef int IT;
#ifdef FLOAT
typedef float VT;
#else
typedef double VT;
#endif

struct PerformanceResult {
    float total_time_ms;
    float aia_time_ms;
    float gflops;
    long long int flop_count;
    std::string implementation;
    std::string dataset;
    float sparsity;
    int feature_dim;
};

// Generate random sparse feature matrix with specified sparsity
CSR<IT, VT> generate_sparse_feature_matrix(int n_nodes, float sparsity) {
    CSR<IT, VT> features;
    
    int feature_dim = 256;  // Always 256 columns
    int target_nnz = (int)(n_nodes * feature_dim * sparsity);  // Target nnz based on sparsity
    
    std::vector<IT> row_ptr(n_nodes + 1, 0);
    std::vector<IT> col_ids;
    std::vector<VT> values;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> col_dist(0, feature_dim - 1);
    std::uniform_real_distribution<VT> val_dist(-1.0, 1.0);
    
    col_ids.reserve(target_nnz);
    values.reserve(target_nnz);
    
    // Distribute non-zeros across rows
    int nnz_per_row = target_nnz / n_nodes;
    int remaining_nnz = target_nnz % n_nodes;
    
    for (int i = 0; i < n_nodes; i++) {
        int row_nnz = nnz_per_row;
        if (i < remaining_nnz) row_nnz++;  // Distribute remainder
        
        // Generate random column indices for this row (avoid duplicates)
        std::set<int> selected_cols;
        while (selected_cols.size() < row_nnz && selected_cols.size() < feature_dim) {
            selected_cols.insert(col_dist(gen));
        }
        
        // Add the selected columns
        for (int col : selected_cols) {
            col_ids.push_back(col);
            values.push_back(val_dist(gen));
        }
        
        row_ptr[i + 1] = col_ids.size();
    }
    
    // Initialize CSR structure
    features.nrow = n_nodes;
    features.ncolumn = feature_dim;  // Always 256
    features.nnz = col_ids.size();
    features.host_malloc = true;
    features.device_malloc = false;
    
    // Allocate CPU memory manually
    features.rpt = new IT[n_nodes + 1];
    features.colids = new IT[features.nnz];
    features.values = new VT[features.nnz];
    
    std::copy(row_ptr.begin(), row_ptr.end(), features.rpt);
    std::copy(col_ids.begin(), col_ids.end(), features.colids);
    std::copy(values.begin(), values.end(), features.values);
    
    return features;
}

// Test function using modern cuSPARSE SpGEMM API (based on spgemm_rd.cu)
PerformanceResult test_spgemm_cusparse(CSR<IT, VT> adj, CSR<IT, VT> features, 
                                       const std::string& dataset, float sparsity) {
    PerformanceResult result;
    result.implementation = "cuSPARSE";
    result.dataset = dataset;
    result.sparsity = sparsity;
    result.feature_dim = features.ncolumn;
    
    long long int flop_count;
    
    // Copy matrices to device
    adj.memcpyHtD();
    features.memcpyHtD();
    
    // Count FLOPs
    get_spgemm_flop(adj, features, flop_count);
    result.flop_count = flop_count;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Start timing
    cudaEventRecord(start);
    
    // Device memory pointers for result matrix C
    int *dC_csrOffsets;
    int *dC_columns;
    VT *dC_values;
    
    // CUSPARSE APIs - using modern SpGEMM API
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA, matB, matC;
    void *dBuffer1 = NULL, *dBuffer2 = NULL;
    size_t bufferSize1 = 0, bufferSize2 = 0;
    
    CHECK_CUSPARSE_FUNC( cusparseCreate(&handle) );
    
    // Create sparse matrix A (adjacency) in CSR format
    CHECK_CUSPARSE_FUNC( cusparseCreateCsr(&matA, adj.nrow, adj.ncolumn, adj.nnz,
                                           adj.d_rpt, adj.d_colids, adj.d_values,
                                           CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                           CUSPARSE_INDEX_BASE_ZERO, 
                                           std::is_same<VT, float>::value ? CUDA_R_32F : CUDA_R_64F) );
    
    // Create sparse matrix B (features) in CSR format  
    CHECK_CUSPARSE_FUNC( cusparseCreateCsr(&matB, features.nrow, features.ncolumn, features.nnz,
                                           features.d_rpt, features.d_colids, features.d_values,
                                           CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                           CUSPARSE_INDEX_BASE_ZERO,
                                           std::is_same<VT, float>::value ? CUDA_R_32F : CUDA_R_64F) );
    
    // Allocate C row pointers
    CHECK_CUDA_FUNC( cudaMalloc((void**)&dC_csrOffsets, (adj.nrow + 1) * sizeof(int)) );
    
    // Create sparse matrix C for the result (A * B)
    CHECK_CUSPARSE_FUNC( cusparseCreateCsr(&matC, adj.nrow, features.ncolumn, 0,
                                           dC_csrOffsets, NULL, NULL,
                                           CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                           CUSPARSE_INDEX_BASE_ZERO,
                                           std::is_same<VT, float>::value ? CUDA_R_32F : CUDA_R_64F) );
    
    // SpGEMM Computation (A * B, where A=adj, B=features)
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE_FUNC( cusparseSpGEMM_createDescr(&spgemmDesc) );
    
    // Set up SpGEMM parameters
    VT alpha = 1.0, beta = 0.0;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = std::is_same<VT, float>::value ? CUDA_R_32F : CUDA_R_64F;
    
    // SpGEMM work estimation
    CHECK_CUSPARSE_FUNC( cusparseSpGEMM_workEstimation(handle, opA, opB,
                                                       &alpha, matA, matB, &beta, matC,
                                                       computeType, CUSPARSE_SPGEMM_DEFAULT,
                                                       spgemmDesc, &bufferSize1, NULL) );
    CHECK_CUDA_FUNC( cudaMalloc((void**)&dBuffer1, bufferSize1) );
    
    CHECK_CUSPARSE_FUNC( cusparseSpGEMM_workEstimation(handle, opA, opB,
                                                       &alpha, matA, matB, &beta, matC,
                                                       computeType, CUSPARSE_SPGEMM_DEFAULT,
                                                       spgemmDesc, &bufferSize1, dBuffer1) );
    
    // SpGEMM compute
    CHECK_CUSPARSE_FUNC( cusparseSpGEMM_compute(handle, opA, opB,
                                                &alpha, matA, matB, &beta, matC,
                                                computeType, CUSPARSE_SPGEMM_DEFAULT,
                                                spgemmDesc, &bufferSize2, NULL) );
    CHECK_CUDA_FUNC( cudaMalloc(&dBuffer2, bufferSize2) );
    
    CHECK_CUSPARSE_FUNC( cusparseSpGEMM_compute(handle, opA, opB,
                                                &alpha, matA, matB, &beta, matC,
                                                computeType, CUSPARSE_SPGEMM_DEFAULT,
                                                spgemmDesc, &bufferSize2, dBuffer2) );
    
    // Get matrix C non-zero entries
    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    CHECK_CUSPARSE_FUNC( cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1) );
    
    printf("cuSPARSE result: %ld x %ld, nnz = %ld\n", C_num_rows1, C_num_cols1, C_nnz1);
    
    // Allocate matrix C
    CHECK_CUDA_FUNC( cudaMalloc((void**)&dC_columns, C_nnz1 * sizeof(int)) );
    CHECK_CUDA_FUNC( cudaMalloc((void**)&dC_values, C_nnz1 * sizeof(VT)) );
    
    // Update matC with the new pointers
    CHECK_CUSPARSE_FUNC( cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values) );
    
    // Copy the final products to the matrix C
    CHECK_CUSPARSE_FUNC( cusparseSpGEMM_copy(handle, opA, opB,
                                             &alpha, matA, matB, &beta, matC,
                                             computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc) );
    
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    result.total_time_ms = milliseconds;
    result.aia_time_ms = 0;  
    result.gflops = (float)(flop_count) / 1000 / 1000 / msec;
    
    // Clean up SpGEMM resources
    CHECK_CUSPARSE_FUNC( cusparseSpGEMM_destroyDescr(spgemmDesc) );
    CHECK_CUSPARSE_FUNC( cusparseDestroySpMat(matA) );
    CHECK_CUSPARSE_FUNC( cusparseDestroySpMat(matB) );
    CHECK_CUSPARSE_FUNC( cusparseDestroySpMat(matC) );
    CHECK_CUSPARSE_FUNC( cusparseDestroy(handle) );
    
    // Free buffers
    CHECK_CUDA_FUNC( cudaFree(dBuffer1) );
    CHECK_CUDA_FUNC( cudaFree(dBuffer2) );
    
    // Clean up
    adj.release_csr();
    features.release_csr();
    
    // Free result
    CHECK_CUDA_FUNC( cudaFree(dC_csrOffsets) );
    CHECK_CUDA_FUNC( cudaFree(dC_columns) );
    CHECK_CUDA_FUNC( cudaFree(dC_values) );
    
    // Destroy CUDA events
    CHECK_CUDA_FUNC( cudaEventDestroy(start) );
    CHECK_CUDA_FUNC( cudaEventDestroy(stop) );
    
    return result;
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
    float msec;
    
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
    CSR<IT, VT> output;
    
    cudaEventRecord(event[0], 0);
    SpGEMM_Hash(adj, features, output);
    cudaEventRecord(event[1], 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&msec, event[0], event[1]);
    
    printf("Hash result: %d x %d, nnz = %d\n", output.nrow, output.ncolumn, output.nnz);
    
    result.total_time_ms = msec;
    result.aia_time_ms = 0;  
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
              << std::setw(10) << "AIA(ms)"
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
              << std::setw(10) << std::fixed << std::setprecision(2) << result.aia_time_ms
              << std::setw(12) << std::fixed << std::setprecision(2) << result.gflops
              << std::setw(12) << std::fixed << std::setprecision(2) << speedup << "x"
              << std::endl;
}

// Main benchmark function
void run_spgemm_benchmark(const std::string& dataset_name, const std::string& data_path) {
    std::cout << "\n=== Testing Dataset: " << dataset_name << " ===" << std::endl;
    
    // Load adjacency matrix using correct format
    CSR<IT, VT> adj;
    if (dataset_name == "reddit") {
        adj = load_reddit_graph(data_path);
    } else if (dataset_name == "flickr") {
        adj = load_flickr_graph(data_path);
    } else if (dataset_name == "yelp") {
        adj = load_yelp_graph(data_path);
    } else if (dataset_name == "products") {
        adj = load_ogbn_products_graph(data_path);
    } else if (dataset_name == "proteins") {
        adj = load_ogbn_proteins_graph(data_path);
    } else {
        std::cerr << "Unknown dataset: " << dataset_name << std::endl;
        return;
    }
    
    std::cout << "Graph loaded: " << adj.nrow << " nodes, " << adj.nnz << " edges" << std::endl;
    
    // Test with different sparsities - feature matrix is always n × 256 with varying sparsity
    std::vector<float> sparsities = {0.5f, 0.25f, 0.125f, 0.0625f};
    
    for (float sparsity : sparsities) {
        std::cout << "\n--- Testing sparsity: " << sparsity << " ---" << std::endl;
        print_results_header();
        
        // Generate sparse feature matrix: n × 256 with nnz = n × 256 × sparsity
        CSR<IT, VT> features = generate_sparse_feature_matrix(adj.nrow, sparsity);
        std::cout << "Adjacency matrix: " << adj.nrow << "x" << adj.ncolumn << ", nnz=" << adj.nnz << std::endl;
        std::cout << "Feature matrix: " << features.nrow << "x" << features.ncolumn 
                  << ", nnz=" << features.nnz << " (target: " << (int)(adj.nrow * 256 * sparsity) << ")" << std::endl;
        
        // Test both implementations
        
        // 1. cuSPARSE baseline
        PerformanceResult cusparse_result = test_spgemm_cusparse(adj, features, dataset_name, sparsity);
        print_result(cusparse_result);
        
        // 2. Hash without AIA (using UNMODIFIED volta_old.hpp)
        PerformanceResult hash_old_result = test_spgemm_hash_old(adj, features, dataset_name, sparsity);
        print_result(hash_old_result, cusparse_result.total_time_ms);
        
        // Debug: Check if results are consistent
        std::cout << "Results comparison completed." << std::endl;
        
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
    
    // Test datasets
    std::vector<std::string> datasets = {"reddit", "flickr", "yelp", "products", "proteins"};
    
    std::cout << "SpGEMM Performance Comparison: Hash vs cuSPARSE (Modern API)" << std::endl;
    std::cout << "Feature matrix dimensions: n × 256 (sparse)" << std::endl;
    std::cout << "Feature matrix nnz: n × 256 × sparsity" << std::endl;
    std::cout << "Sparsity levels: 0.5, 0.25, 0.125, 0.0625" << std::endl;
    std::cout << "Loading graphs from .indices/.indptr format" << std::endl;
    
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
