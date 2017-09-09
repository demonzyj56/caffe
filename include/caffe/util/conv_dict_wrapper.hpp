#ifndef CAFFE_CONV_DICT_WRAPPER_HPP_
#define CAFFE_CONV_DICT_WRAPPER_HPP_

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "cusparse.h"
// #include "cusolverDn.h"
// #include "cusolverSp.h"

namespace caffe {

template <typename Dtype>
void make_conv_dict_cpu(const int n, const int m, const Dtype *Dl, const int N, 
    CSCParameter::Boundary boundary, Dtype *values, int *columns, int *ptrB);
template <typename Dtype>
void make_conv_dict_gpu(const int n, const int m, const Dtype *Dl, const int N, 
    CSCParameter::Boundary boundary, Dtype *values, int *columns, int *ptrB);

#define CUSPARSE_CHECK(condition) do{ \
    cusparseStatus_t status = condition; \
    CHECK_EQ(status, CUSPARSE_STATUS_SUCCESS) \
        << "cusparse fail!"; \
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

private:
    cusparseHandle_t *handle_;
    cusparseMatDescr_t descr_;
    int r_;
    int c_;
    int nnz_;
    Dtype *d_values_;
    int *d_columns_;
    int *d_ptrB_;

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

    ~ConvDictWrapper();

    // solve (D^tD + lambda2*I)beta = x.
    // Input should be on device side.
    // The output will overwrite d_x.
    void solve(int nnz, const int *d_inds, Dtype *d_x);

    // accessor
    shared_ptr<CSRWrapper<Dtype> > D() const { return D_; }
    shared_ptr<CSRWrapper<Dtype> > DtDpl2I() const { return DtDpl2I_; }

    // Create the base matrix (DtD+lambda2*I).
    // Use the cusparseScsrgemm2 routine.
    void create();


private:
    cusparseHandle_t *handle_;
    int n_;
    int m_;
    int N_;
    CSCParameter::Boundary boundary_;
    Dtype lambda2_;
    shared_ptr<CSRWrapper<Dtype> > D_;
    shared_ptr<CSRWrapper<Dtype> > DtDpl2I_;

    DISABLE_COPY_AND_ASSIGN(ConvDictWrapper);

};

} // namespace caffe


#endif // CAFFE_CONV_DICT_WRAPPER_HPP_
