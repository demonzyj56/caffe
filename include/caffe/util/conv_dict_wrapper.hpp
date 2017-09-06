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

// template <typename Dtype>
// class CSRWrapper {
// public:
//     CSRWrapper();
//     explicit CSRWrapper(cusparseHandle_t handle, int r, int c, int nnz,
//         const Dtype* values, int *columns, int *ptrB);
//     ~CSRWrapper();
//
//     int row() const { return r_; }
//     int col() const { return c_; }
//     int nnz() const { return nnz_; }
//
// private:
//     cusparseHandle_t handle_;
//     int r_;
//     int c_;
//     int nnz_;
//     Dtype *d_values_;
//     int *columns_;
//     int *ptrB_;
// };

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
 * A wrapper to create the convolutional dictionary from
 * local dictionary $Dl$.
 * 
 * The convolutional dictionary is a zero-based sparse CSR format matrix.
 * 
 * 
 * */
// template <typename Dtype>
// class ConvDictWrapper {
// public:
//     explicit ConvDictWrapper(const Blob<Dtype> *Dl, int N,
//         CSCParameter::Boundary boundary, Dtype lambda2);
//
//     ~ConvDictWrapper();
//
//     // solve (D^tD + lambda2*I)beta = x.
//     // Input should be on device side.
//     // The output will overwrite d_x.
//     void solve(int nnz, const int *d_inds, Dtype *d_x);
//
// protected:
//     void make_conv_dict();
//
// private:
//     Dtype *h_values_;
//     int *h_columns_;
//     int *h_ptrB_;
//     int *h_ptrE_;
//     Dtype lambda2_;
//     CSCParameter::Boundary boundary_;
//     int n_;
//     int m_;
//
// DISABLE_COPY_AND_ASSIGN;
//
// };

} // namespace caffe


#endif // CAFFE_CONV_DICT_WRAPPER_HPP_
