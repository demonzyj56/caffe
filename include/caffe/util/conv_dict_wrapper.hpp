#ifndef CAFFE_CONV_DICT_WRAPPER_HPP_
#define CAFFE_CONV_DICT_WRAPPER_HPP_

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "cusparse.h"
#include "cusolverDn.h"  // for computing eigenvalues
#include "cusolverSp.h"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void make_conv_dict_cpu(const int n, const int m, const Dtype *Dl, const int N, 
    CSCParameter::Boundary boundary, Dtype *values, int *columns, int *ptrB);
template <typename Dtype>
void make_conv_dict_gpu(const int n, const int m, const Dtype *Dl, const int N, 
    CSCParameter::Boundary boundary, Dtype *values, int *columns, int *ptrB);

template <typename Dtype>
void make_transposed_conv_dict_cpu(int n, int m, const Dtype *Dl, int channels, int height, int width,
        int kernel_h, int kernel_w, CSCParameter::Boundary boundary, Dtype *values, int *columns,
        int *ptrB);
template <typename Dtype>
void make_transposed_conv_dict_gpu(int n, int m, const Dtype *Dl, int channels, int height, int width,
        int kernel_h, int kernel_w, CSCParameter::Boundary boundary, Dtype *values, int *columns,
        int *ptrB);

inline std::string cusolverGetErrorString(cusolverStatus_t status) {
    switch (status) {
        case CUSOLVER_STATUS_SUCCESS:
            return "cusolver status success";
        case CUSOLVER_STATUS_NOT_INITIALIZED:
            return "cusolver status not initialized";
        case CUSOLVER_STATUS_ALLOC_FAILED:
            return "cusolver status alloc failed";
        case CUSOLVER_STATUS_INVALID_VALUE:
            return "cusolver status invalid value";
        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "cusolver status arch mismatch";
        case CUSOLVER_STATUS_EXECUTION_FAILED:
            return "cusolver status execution failed";
        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "cusolver status internal error";
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "cusolver status matrix type not supported";
        default:
            return "Unknown cusolver error type.";
    }
}

#define CUSPARSE_CHECK(condition) do{ \
    cusparseStatus_t status = condition; \
    CHECK_EQ(status, CUSPARSE_STATUS_SUCCESS) \
        << "cusparse fail!"; \
} while(0)

#define CUSOLVER_CHECK(condition) do{ \
    cusolverStatus_t status = condition; \
    CHECK_EQ(status, CUSOLVER_STATUS_SUCCESS) \
        << cusolverGetErrorString(status); \
} while(0)

/*
 * A wrapper for the cusparse handle.
 * Notice the program dies as long as creating the handle is not successful.
 * This is to remind that the handle should be created only when needed.
 * */
class CusparseHandle {
public:
    CusparseHandle() : handle_(NULL) {
        CHECK_EQ(CUSPARSE_STATUS_SUCCESS, cusparseCreate(&handle_))
            << "Unable to create cusparse handle!";
    }
    ~CusparseHandle() {
        cusparseStatus_t status = cusparseDestroy(handle_);
        LOG_IF(INFO, CUSPARSE_STATUS_SUCCESS != status)
            << "Destroying the cusparse handle is not successful!";
        LOG_IF(INFO, CUSPARSE_STATUS_NOT_INITIALIZED == status)
            << "The handle is not initialized!";
    }
    cusparseHandle_t *get() {
        return &handle_;
    }

private:
    cusparseHandle_t handle_;

    DISABLE_COPY_AND_ASSIGN(CusparseHandle);
};


/*
 * A wrapper for cusparse matrix.
 * The matrix is zero-based and of csr format.
 * The memory are all located at device side.
 * Note that the row and column of the matrix are assumed to be invariant,
 * while the nnz could vary.
 * */
template <typename Dtype>
class CSRWrapper {
public:
    explicit CSRWrapper(cusparseHandle_t *handle, int r, int c, int nnz);
    ~CSRWrapper();

    int row() const { return r_; }
    int col() const { return c_; }
    int nnz() const { return nnz_; }
    const Dtype *values()  { return mutable_values(); }
    const int   *columns() { return mutable_columns(); }
    const int   *ptrB()    { return mutable_ptrB(); }
    const int   *ptrE()    { return mutable_ptrE(); }
    Dtype *mutable_values(); 
    int   *mutable_columns();
    int   *mutable_ptrB();
    int   *mutable_ptrE() { return mutable_ptrB() + 1; }
    const Dtype *cpu_values()  { return mutable_cpu_values(); }
    const int   *cpu_columns() { return mutable_cpu_columns(); }
    const int   *cpu_ptrB()    { return mutable_cpu_ptrB(); }
    const int   *cpu_ptrE()    { return mutable_cpu_ptrE(); }
    Dtype *mutable_cpu_values(); 
    int   *mutable_cpu_columns();
    int   *mutable_cpu_ptrB();
    int   *mutable_cpu_ptrE() { return mutable_cpu_ptrB() + 1; }
    void set_nnz(int nnz);
    void set_values(const Dtype *values);
    void set_columns(const int *columns);
    void set_ptrB(const int *ptrB);
    cusparseMatDescr_t descr() { return descr_; }
    void set_matrix_type(cusparseMatrixType_t cusparse_matrix_type);
    void set_fill_mode(cusparseFillMode_t cusparse_fill_mode);
    void set_diag_type(cusparseDiagType_t cusparse_diag_type);
    void set_index_base(cusparseIndexBase_t cusparse_index_base);
    shared_ptr<CSRWrapper<Dtype> > transpose();
    CSRWrapper<Dtype> &identity();
    void to_dense(Dtype *d_dense);
    // Remove and keep only indices appeared in inds.
    shared_ptr<CSRWrapper<Dtype> > clip(int nnz, const int *h_inds);
    // Remove the columns that appeared in inds.
    shared_ptr<CSRWrapper<Dtype> > clip_columns(int nnz, const int *h_inds);
    shared_ptr<CSRWrapper<Dtype> > clip_columns_gpu(int nnz, const int *inds) {
        SyncedMemory d_inds(sizeof(int)*nnz);
        caffe_copy(nnz, inds, (int *)d_inds.mutable_gpu_data());
        return clip_columns_gpu_(nnz, (const int *)d_inds.gpu_data());
    }
    bool symmetric();
    // sort the values and columns inplace
    void sort();
    // Typically we put a negative integer at invalid entries in columns().
    // Prune all invalid entries and adjust nnz.
    void prune();

protected:
    // Impl on device side.  Note the indices value is also on device side.
    shared_ptr<CSRWrapper<Dtype> > clip_columns_gpu_(int nnz, const int *d_inds);

private:
    cusparseHandle_t *handle_;
    cusparseMatDescr_t descr_;
    int r_;
    int c_;
    int nnz_;
    shared_ptr<SyncedMemory> d_values_;
    shared_ptr<SyncedMemory> d_columns_;
    shared_ptr<SyncedMemory> d_ptrB_;

    DISABLE_COPY_AND_ASSIGN(CSRWrapper);
};

/*
 * A wrapper to create the convolutional dictionary from
 * local dictionary $Dl$.
 * 
 * The convolutional dictionary is a zero-based sparse CSR format matrix.
 * 
 * */
template <typename Dtype>
class ConvDictWrapper {
public:
    explicit ConvDictWrapper(cusparseHandle_t *handle, const Blob<Dtype> *Dl, int N,
        CSCParameter::Boundary boundary, Dtype lambda2);

    explicit ConvDictWrapper(cusparseHandle_t *handle, const Blob<Dtype> *Dl, int channels,
        int height, int width, int kernel_h, int kernel_w, CSCParameter::Boundary boundary,
        Dtype lambda2);

    ~ConvDictWrapper();

    // create D_, using GPU routines
    void make_conv_dict(const Blob<Dtype> *Dl);

    // solve (D^tD + lambda2*I)beta = x.
    // The index should be on host side while the output is on device side.
    // The output will overwrite d_x.
    void solve(int nnz, const int *h_inds, Dtype *d_x);

    // Analyse the pattern given by h_inds.
    // Check whether the matrix created using h_inds is positive definite.
    // This is equivalent to checking whether the matrix given by clip
    // is symmetric, and whether all the eigenvalues are larger than zero.
    void analyse(int nnz, const int *h_inds);

    // accessor
    shared_ptr<CSRWrapper<Dtype> > D() const { return D_; }
    shared_ptr<CSRWrapper<Dtype> > DtDpl2I() const { return DtDpl2I_; }

    // Create the base matrix (DtD+lambda2*I).
    // Use the cusparseScsrgemm2 routine.
    void create();

    // Create the clipped matrix using the indices.
    // This prevents from creating a large matrix, but requires
    // matrix multiplication for each step.
    shared_ptr<CSRWrapper<Dtype> > create_clipped(int nnz, const int *h_inds);

private:
    cusparseHandle_t *handle_;
    cusolverDnHandle_t dnsolver_handle_;
    cusolverSpHandle_t spsolver_handle_;
    int n_;
    int m_;
    int N_;
    CSCParameter::Boundary boundary_;
    Dtype lambda2_;
    shared_ptr<CSRWrapper<Dtype> > D_;
    shared_ptr<CSRWrapper<Dtype> > DtDpl2I_;
    bool debug_;
    int channels_;
    int height_;
    int width_;
    int kernel_h_;
    int kernel_w_;

    DISABLE_COPY_AND_ASSIGN(ConvDictWrapper);

};

} // namespace caffe


#endif // CAFFE_CONV_DICT_WRAPPER_HPP_
