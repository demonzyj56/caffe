#include "caffe/util/conv_dict_wrapper.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void make_conv_dict_cpu(const int n, const int m, const Dtype *Dl, const int N,
    CSCParameter::Boundary boundary, Dtype *values, int *columns, int *ptrB) {
    CHECK_EQ(boundary, CSCParameter::CIRCULANT_BACK) 
        << "Only circulant back boundary is supported!";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < n; ++j) {
            int block = j > i ? j - n : j;
            int ind = (block + N) % N;
            int r = (n - j + i) % n;
            for (int k = 0; k < m; ++k) {
                *columns = ind * m + k;
                *values = Dl[r*m + k];
                columns++;
                values++;
            }
        }
    }
    for (int i = 0; i <= N; ++i) {
        ptrB[i] = i * n * m;
    }
}

template void make_conv_dict_cpu<float>(const int n, const int m, const float *Dl, const int N,
    CSCParameter::Boundary boundary, float *values, int *columns, int *ptrB);
template void make_conv_dict_cpu<double>(const int n, const int m, const double *Dl, const int N,
    CSCParameter::Boundary boundary, double *values, int *columns, int *ptrB);

template <typename Dtype>
CSRWrapper<Dtype>::CSRWrapper(cusparseHandle_t *handle, int r, int c, int nnz)
    : handle_(handle), descr_(NULL), r_(r), c_(c), nnz_(nnz),
    d_values_(NULL), d_columns_(NULL), d_ptrB_(NULL) {
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descr_));
    CUSPARSE_CHECK(cusparseSetMatType(descr_, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descr_, CUSPARSE_INDEX_BASE_ZERO));  // zero based
}

template <typename Dtype>
CSRWrapper<Dtype>::~CSRWrapper() {
    CUSPARSE_CHECK(cusparseDestroyMatDescr(descr_));
    if (NULL != d_values_) {
        CUDA_CHECK(cudaFree(d_values_));
    }
    if (NULL != d_columns_) {
        CUDA_CHECK(cudaFree(d_columns_));
    }
    if (NULL != d_ptrB_) {
        CUDA_CHECK(cudaFree(d_ptrB_));
    }
}

template <typename Dtype>
Dtype *CSRWrapper<Dtype>::mutable_values() {
    CHECK(nnz_ >= 0);
    if (NULL == d_values_) {
        CUDA_CHECK(cudaMalloc((void**)&d_values_, sizeof(Dtype)*nnz_));
    }
    return d_values_;
}

template <typename Dtype>
int *CSRWrapper<Dtype>::mutable_columns() {
    CHECK(nnz_ >= 0);
    if (NULL == d_columns_) {
        CUDA_CHECK(cudaMalloc((void**)&d_columns_, sizeof(int)*nnz_));
    }
    return d_columns_;
}

template <typename Dtype>
int *CSRWrapper<Dtype>::mutable_ptrB() {
    CHECK(r_ > 0);
    if (NULL == d_ptrB_) {
        CUDA_CHECK(cudaMalloc((void**)&d_ptrB_, sizeof(int)*(r_+1)));
    }
    return d_ptrB_;
}

template <typename Dtype>
void CSRWrapper<Dtype>::set_nnz(int nnz) {
    if (nnz_ != nnz) {
        if (NULL != d_values_) {
            CUDA_CHECK(cudaFree(d_values_));
            d_values_ = NULL;
        }
        if (NULL != d_columns_) {
            CUDA_CHECK(cudaFree(d_columns_));
            d_columns_ = NULL;
        }
    }
    nnz_ = nnz;
}

template <typename Dtype>
void CSRWrapper<Dtype>::set_values(const Dtype *values) {
    CHECK(nnz_ >= 0);
    caffe_copy(nnz_, values, mutable_values());
}

template <typename Dtype>
void CSRWrapper<Dtype>::set_columns(const int *columns) {
    CHECK(nnz_ >= 0);
    caffe_copy(nnz_, columns, mutable_columns());
}

template <typename Dtype>
void CSRWrapper<Dtype>::set_ptrB(const int *ptrB) {
    CHECK(r_ > 0);
    caffe_copy(r_+1, ptrB, mutable_ptrB());
}

template <typename Dtype>
void CSRWrapper<Dtype>::set_matrix_type(cusparseMatrixType_t cusparse_matrix_type) {
    CUSPARSE_CHECK(cusparseSetMatType(descr_, cusparse_matrix_type));
}

template <typename Dtype>
void CSRWrapper<Dtype>::set_fill_mode(cusparseFillMode_t cusparse_fill_mode) {
    CUSPARSE_CHECK(cusparseSetMatFillMode(descr_, cusparse_fill_mode));
}

template <typename Dtype>
void CSRWrapper<Dtype>::set_diag_type(cusparseDiagType_t cusparse_diag_type) {
    CUSPARSE_CHECK(cusparseSetMatDiagType(descr_, cusparse_diag_type));
}

template <typename Dtype>
void CSRWrapper<Dtype>::set_index_base(cusparseIndexBase_t cusparse_index_base) {
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descr_, cusparse_index_base));
}

template <>
shared_ptr<CSRWrapper<float> > CSRWrapper<float>::transpose() {
    shared_ptr<CSRWrapper<float> > Dtrans = shared_ptr<CSRWrapper<float> >(
        new CSRWrapper(handle_, col(), row(), nnz()));
    CUSPARSE_CHECK(cusparseScsr2csc(
        *handle_, row(), col(), nnz(),
        values(), ptrB(), columns(),
        Dtrans->mutable_values(),
        Dtrans->mutable_columns(),
        Dtrans->mutable_ptrB(),
        CUSPARSE_ACTION_NUMERIC,
        cusparseGetMatIndexBase(descr())
        ));
    return Dtrans;
}

template <>
shared_ptr<CSRWrapper<double> > CSRWrapper<double>::transpose() {
    shared_ptr<CSRWrapper<double> > Dtrans = shared_ptr<CSRWrapper<double> >(
        new CSRWrapper(handle_, col(), row(), nnz()));
    CUSPARSE_CHECK(cusparseDcsr2csc(
        *handle_, row(), col(), nnz(),
        values(), ptrB(), columns(),
        Dtrans->mutable_values(),
        Dtrans->mutable_columns(),
        Dtrans->mutable_ptrB(),
        CUSPARSE_ACTION_NUMERIC,
        cusparseGetMatIndexBase(descr())
        ));
    return Dtrans;
}

template <typename Dtype>
ConvDictWrapper<Dtype>::ConvDictWrapper(cusparseHandle_t *handle, const Blob<Dtype> *Dl, int N,
    CSCParameter::Boundary boundary, Dtype lambda2)
    : handle_(handle), n_(Dl->shape(0)), m_(Dl->shape(1)), N_(N),
    boundary_(boundary), lambda2_(lambda2),
    D_(new CSRWrapper<Dtype>(handle, N_, N_*m_, n_*m_*N_)),
    DtDpl2I_(new CSRWrapper<Dtype>(handle, N_*m_, N_*m_, -1)) {
    CHECK_EQ(boundary_, CSCParameter::CIRCULANT_BACK)
        << "Only circulant back boundary condtion is currently supported!";
    make_conv_dict_gpu(n_, m_, Dl->gpu_data(), N_, boundary_,
        D_->mutable_values(), D_->mutable_columns(), D_->mutable_ptrB());
}

template <typename Dtype>
ConvDictWrapper<Dtype>::~ConvDictWrapper() {
}

template <>
void ConvDictWrapper<float>::create() {
    int base, nnz;
    csrgemm2Info_t info = NULL;
    size_t bufferSize;
    int *nnzTotalDevHostPtr = &nnz;
    float alpha = 1.;
    float beta = lambda2_;
    CUSPARSE_CHECK(cusparseSetPointerMode(*handle_, CUSPARSE_POINTER_MODE_HOST));
    // step 0: create transposed D and identity matrix and output
    shared_ptr<CSRWrapper<float> > Dtrans = D_->transpose();
    shared_ptr<CSRWrapper<float> > identity = shared_ptr<CSRWrapper<float> >(
        new CSRWrapper<float>(handle_, D_->col(), D_->col(), D_->col()));
    identity->identity();
    // step 1: create an opaque structure
    CUSPARSE_CHECK(cusparseCreateCsrgemm2Info(&info));
    // step 2: allocate buffer for csrgemm2Nnz and csrgemm2
    CUSPARSE_CHECK(cusparseScsrgemm2_bufferSizeExt(
        *handle_, D_->col(), D_->col(), D_->row(), &alpha,
        Dtrans->descr(), Dtrans->nnz(), Dtrans->ptrB(), Dtrans->columns(),
        D_->descr(), D_->nnz(), D_->ptrB(), D_->columns(),
        &beta,
        identity->descr(), identity->nnz(), identity->ptrB(), identity->columns(),
        info, &bufferSize));
    SyncedMemory buffer(bufferSize);
    //step 3: compute csrRowPtrC
    CUSPARSE_CHECK(cusparseXcsrgemm2Nnz(*handle_, D_->col(), D_->col(), D_->row(),
        Dtrans->descr(), Dtrans->nnz(), Dtrans->ptrB(), Dtrans->columns(),
        D_->descr(), D_->nnz(), D_->ptrB(), D_->columns(),
        identity->descr(), identity->nnz(), identity->ptrB(), identity->columns(),
        DtDpl2I_->descr(), DtDpl2I_->mutable_ptrB(), nnzTotalDevHostPtr,
        info, buffer.mutable_gpu_data()));
    if (NULL != nnzTotalDevHostPtr) {
        nnz = *nnzTotalDevHostPtr;
    } else {
        CUDA_CHECK(cudaMemcpy(&nnz, DtDpl2I_->ptrB()+DtDpl2I_->row(), sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&base, DtDpl2I_->ptrB(), sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_EQ(base, 0);
        nnz -= base;
    }
    DtDpl2I_->set_nnz(nnz);
    // step 4: finith sparsity pattern and value of C
    CUSPARSE_CHECK(cusparseScsrgemm2(*handle_, D_->col(), D_->col(), D_->row(), &alpha,
        Dtrans->descr(), Dtrans->nnz(), Dtrans->values(), Dtrans->ptrB(), Dtrans->columns(),
        D_->descr(), D_->nnz(), D_->values(), D_->ptrB(), D_->columns(),
        &beta,
        identity->descr(), identity->nnz(), identity->values(), identity->ptrB(), identity->columns(),
        DtDpl2I_->descr(), DtDpl2I_->mutable_values(), DtDpl2I_->mutable_ptrB(), DtDpl2I_->mutable_columns(),
        info, buffer.mutable_gpu_data()));
    // step 5: destroy and free memory
    CUSPARSE_CHECK(cusparseDestroyCsrgemm2Info(info));
}

template <>
void ConvDictWrapper<double>::create() {
    int base, nnz;
    csrgemm2Info_t info = NULL;
    size_t bufferSize;
    int *nnzTotalDevHostPtr = &nnz;
    double alpha = 1.;
    double beta = lambda2_;
    CUSPARSE_CHECK(cusparseSetPointerMode(*handle_, CUSPARSE_POINTER_MODE_HOST));
    // step 0: create transposed D and identity matrix and output
    shared_ptr<CSRWrapper<double> > Dtrans = D_->transpose();
    shared_ptr<CSRWrapper<double> > identity = shared_ptr<CSRWrapper<double> >(
        new CSRWrapper<double>(handle_, D_->col(), D_->col(), D_->col()));
    identity->identity();
    // step 1: create an opaque structure
    CUSPARSE_CHECK(cusparseCreateCsrgemm2Info(&info));
    // step 2: allocate buffer for csrgemm2Nnz and csrgemm2
    CUSPARSE_CHECK(cusparseDcsrgemm2_bufferSizeExt(
        *handle_, D_->col(), D_->col(), D_->row(), &alpha,
        Dtrans->descr(), Dtrans->nnz(), Dtrans->ptrB(), Dtrans->columns(),
        D_->descr(), D_->nnz(), D_->ptrB(), D_->columns(),
        &beta,
        identity->descr(), identity->nnz(), identity->ptrB(), identity->columns(),
        info, &bufferSize));
    SyncedMemory buffer(bufferSize);
    //step 3: compute csrRowPtrC
    CUSPARSE_CHECK(cusparseXcsrgemm2Nnz(*handle_, D_->col(), D_->col(), D_->row(),
        Dtrans->descr(), Dtrans->nnz(), Dtrans->ptrB(), Dtrans->columns(),
        D_->descr(), D_->nnz(), D_->ptrB(), D_->columns(),
        identity->descr(), identity->nnz(), identity->ptrB(), identity->columns(),
        DtDpl2I_->descr(), DtDpl2I_->mutable_ptrB(), nnzTotalDevHostPtr,
        info, buffer.mutable_gpu_data()));
    if (NULL != nnzTotalDevHostPtr) {
        nnz = *nnzTotalDevHostPtr;
    } else {
        CUDA_CHECK(cudaMemcpy(&nnz, DtDpl2I_->ptrB()+DtDpl2I_->row(), sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&base, DtDpl2I_->ptrB(), sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_EQ(base, 0);
        nnz -= base;
    }
    DtDpl2I_->set_nnz(nnz);
    // step 4: finith sparsity pattern and value of C
    CUSPARSE_CHECK(cusparseDcsrgemm2(*handle_, D_->col(), D_->col(), D_->row(), &alpha,
        Dtrans->descr(), Dtrans->nnz(), Dtrans->values(), Dtrans->ptrB(), Dtrans->columns(),
        D_->descr(), D_->nnz(), D_->values(), D_->ptrB(), D_->columns(),
        &beta,
        identity->descr(), identity->nnz(), identity->values(), identity->ptrB(), identity->columns(),
        DtDpl2I_->descr(), DtDpl2I_->mutable_values(), DtDpl2I_->mutable_ptrB(), DtDpl2I_->mutable_columns(),
        info, buffer.mutable_gpu_data()));
    // step 5: destroy and free memory
    CUSPARSE_CHECK(cusparseDestroyCsrgemm2Info(info));
}
/*
template <>
void ConvDictWrapper<float>::create() {
    // need to estimate nnz first
    int base, nnz;
    int *nnzTotalDevHostPtr = &nnz;
    CUSPARSE_CHECK(cusparseSetPointerMode(*handle_, CUSPARSE_POINTER_MODE_HOST));
    CUSPARSE_CHECK(cusparseXcsrgemmNnz(
        *handle_,
        CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        m_*N_,
        m_*N_,
        N_,
        D_->descr(),
        D_->nnz(),
        D_->ptrB(),
        D_->columns(),
        D_->descr(),
        D_->nnz(),
        D_->ptrB(),
        D_->columns(),
        DtDpl2I_->descr(),
        DtDpl2I_->mutable_ptrB(),
        nnzTotalDevHostPtr));
    if (NULL != nnzTotalDevHostPtr) {
        nnz = *nnzTotalDevHostPtr;
    } else {
        CUDA_CHECK(cudaMemcpy(&nnz, DtDpl2I_->mutable_ptrB()+N_*m_, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&base, DtDpl2I_->mutable_ptrB(), sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_EQ(base, 0);
        nnz -= base;
    }
    DtDpl2I_.reset(new CSRWrapper<float>(handle_, N_*m_, N_*m_, nnz));
    CUSPARSE_CHECK(cusparseScsrgemm(
        *handle_,
        CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        m_*N_,
        m_*N_,
        N_,
        D_->descr(),
        D_->nnz(),
        D_->values(),
        D_->ptrB(),
        D_->columns(),
        D_->descr(),
        D_->nnz(),
        D_->values(),
        D_->ptrB(),
        D_->columns(),
        DtDpl2I_->descr(),
        DtDpl2I_->mutable_values(),
        DtDpl2I_->mutable_ptrB(),
        DtDpl2I_->mutable_columns()));
}

template <>
void ConvDictWrapper<double>::create() {
    int base, nnz;
    int *nnzTotalDevHostPtr = &nnz;
    CUSPARSE_CHECK(cusparseSetPointerMode(*handle_, CUSPARSE_POINTER_MODE_HOST));
    CUSPARSE_CHECK(cusparseXcsrgemmNnz(
        *handle_,
        CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        m_*N_,
        m_*N_,
        N_,
        D_->descr(),
        D_->nnz(),
        D_->ptrB(),
        D_->columns(),
        D_->descr(),
        D_->nnz(),
        D_->ptrB(),
        D_->columns(),
        DtDpl2I_->descr(),
        DtDpl2I_->mutable_ptrB(),
        nnzTotalDevHostPtr));
    if (NULL != nnzTotalDevHostPtr) {
        nnz = *nnzTotalDevHostPtr;
    } else {
        CUDA_CHECK(cudaMemcpy(&nnz, DtDpl2I_->mutable_ptrB()+N_*m_, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&base, DtDpl2I_->mutable_ptrB(), sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_EQ(base, 0);
        nnz -= base;
    }
    DtDpl2I_.reset(new CSRWrapper<double>(handle_, N_*m_, N_*m_, nnz));
    CUSPARSE_CHECK(cusparseDcsrgemm(
        *handle_,
        CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        m_*N_,
        m_*N_,
        N_,
        D_->descr(),
        D_->nnz(),
        D_->values(),
        D_->ptrB(),
        D_->columns(),
        D_->descr(),
        D_->nnz(),
        D_->values(),
        D_->ptrB(),
        D_->columns(),
        DtDpl2I_->descr(),
        DtDpl2I_->mutable_values(),
        DtDpl2I_->mutable_ptrB(),
        DtDpl2I_->mutable_columns()));
}
*/

template <typename Dtype>
void ConvDictWrapper<Dtype>::solve(int nnz, const int *d_inds, Dtype *d_x) {
    NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(CSRWrapper);
INSTANTIATE_CLASS(ConvDictWrapper);

} // namespace caffe
