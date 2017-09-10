#include "caffe/util/conv_dict_wrapper.hpp"
#include "caffe/util/math_functions.hpp"
#include <algorithm>

namespace caffe {


template <typename Dtype>
void make_conv_dict_cpu(const int n, const int m, const Dtype *Dl, const int N,
    CSCParameter::Boundary boundary, Dtype *values, int *columns, int *ptrB) {
    CHECK_EQ(boundary, CSCParameter::CIRCULANT_BACK) 
        << "Only circulant back boundary is supported!";
    for (int i = 0; i < N; ++i) {
        int i_prime = i > (n-1) ? (n-1) : i;
        for (int j = 0; j < n; ++j) {
            int row = (i_prime - j + n) % n;
            for (int k = 0; k < m; ++k) {
                values[i*n*m + j*m + k] = Dl[row*m+k];
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int block = (i >= j ? j : j - n + N);
            for (int k = 0; k < m; ++k) {
                columns[i*m*n + j*m + k] = block * m + k;
            }
        }
    }
    for (int i = n; i < N; ++i) {
        for (int j = 0; j < n; ++j) {
            int block = j + i - n + 1;
            for (int k = 0; k < m; ++k) {
                columns[i*m*n + j*m + k] = block * m + k;
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
    d_values_(), d_columns_(), d_ptrB_() {
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descr_));
    CUSPARSE_CHECK(cusparseSetMatType(descr_, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descr_, CUSPARSE_INDEX_BASE_ZERO));  // zero based
}

template <typename Dtype>
CSRWrapper<Dtype>::~CSRWrapper() {
    CUSPARSE_CHECK(cusparseDestroyMatDescr(descr_));
}

template <typename Dtype>
Dtype *CSRWrapper<Dtype>::mutable_values() {
    CHECK(nnz_ >= 0);
    if (NULL == d_values_) {
        d_values_ = shared_ptr<SyncedMemory>(new SyncedMemory(sizeof(Dtype)*nnz_));
    }
    return (Dtype *)d_values_->mutable_gpu_data();
}

template <typename Dtype>
int *CSRWrapper<Dtype>::mutable_columns() {
    CHECK(nnz_ >= 0);
    if (NULL == d_columns_) {
        d_columns_ = shared_ptr<SyncedMemory>(new SyncedMemory(sizeof(int)*nnz_));
    }
    return (int *)d_columns_->mutable_gpu_data();
}

template <typename Dtype>
int *CSRWrapper<Dtype>::mutable_ptrB() {
    CHECK(r_ > 0);
    if (NULL == d_ptrB_) {
        d_ptrB_ = shared_ptr<SyncedMemory>(new SyncedMemory(sizeof(int)*(r_+1)));
    }
    return (int *)d_ptrB_->mutable_gpu_data();
}

template <typename Dtype>
Dtype *CSRWrapper<Dtype>::mutable_cpu_values() {
    CHECK(nnz_ >= 0);
    if (NULL == d_values_) {
        d_values_ = shared_ptr<SyncedMemory>(new SyncedMemory(sizeof(Dtype)*nnz_));
    }
    return (Dtype *)d_values_->mutable_cpu_data();
}

template <typename Dtype>
int *CSRWrapper<Dtype>::mutable_cpu_columns() {
    CHECK(nnz_ >= 0);
    if (NULL == d_columns_) {
        d_columns_ = shared_ptr<SyncedMemory>(new SyncedMemory(sizeof(int)*nnz_));
    }
    return (int *)d_columns_->mutable_cpu_data();
}

template <typename Dtype>
int *CSRWrapper<Dtype>::mutable_cpu_ptrB() {
    CHECK(r_ > 0);
    if (NULL == d_ptrB_) {
        d_ptrB_ = shared_ptr<SyncedMemory>(new SyncedMemory(sizeof(int)*(r_+1)));
    }
    return (int *)d_ptrB_->mutable_cpu_data();
}

template <typename Dtype>
void CSRWrapper<Dtype>::set_nnz(int nnz) {
    if (nnz_ != nnz) {
        if (NULL != d_values_) {
            d_values_.reset();
        }
        if (NULL != d_columns_) {
            d_columns_.reset();
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

// The length of the dense ptr should be r_*c_, and is row-major.
// It is on the device side.
// Note: one tricky thing is that cusparse library represents dense matrix
// in column major, so we have to transpose the csr matrix to a csc matrix
// then return the transposed dense matrix as the row view of the origin one.
template <>
void CSRWrapper<float>::to_dense(float *d_dense) {
    shared_ptr<CSRWrapper<float> > trans = transpose();
    CUSPARSE_CHECK(cusparseScsr2dense(*handle_, col(), row(), trans->descr(),
        trans->values(), trans->ptrB(), trans->columns(), d_dense, col()));
}

template <>
void CSRWrapper<double>::to_dense(double *d_dense) {
    shared_ptr<CSRWrapper<double> > trans = transpose();
    CUSPARSE_CHECK(cusparseDcsr2dense(*handle_, col(), row(), trans->descr(),
        trans->values(), trans->ptrB(), trans->columns(), d_dense, col()));
}

//! naive impl
template <typename Dtype>
shared_ptr<CSRWrapper<Dtype> > CSRWrapper<Dtype>::clip(int nnz, const int *inds) {
    shared_ptr<CSRWrapper<Dtype> > clipped(new CSRWrapper<Dtype>(handle_, nnz, nnz, -1));
    clipped->mutable_cpu_ptrB()[0] = 0;
    for (int i = 0; i < nnz; ++i) {
        int ind = inds[i];
        clipped->mutable_cpu_ptrB()[i+1] = clipped->mutable_cpu_ptrB()[i];
        for (int j = cpu_ptrB()[ind]; j < cpu_ptrE()[ind]; ++j) {
            int target = cpu_columns()[j];
            for (int k = 0; k < nnz; ++k) {
                if (inds[k] == target) {
                    clipped->mutable_cpu_ptrB()[i+1] += 1;
                    break;
                }
            }
        }
    }
    clipped->set_nnz(clipped->cpu_ptrB()[nnz]);
    Dtype *clipped_cpu_values = clipped->mutable_cpu_values();
    int *clipped_cpu_columns = clipped->mutable_cpu_columns();
    for (int i = 0; i < nnz; ++i) {
        int ind = inds[i];
        for (int j = cpu_ptrB()[ind]; j < cpu_ptrE()[ind]; ++j) {
            Dtype target_value = cpu_values()[j];
            int target_column = cpu_columns()[j];
            for (int k = 0; k < nnz; ++k) {
                if (inds[k] == target_column) {
                    *clipped_cpu_values = target_value;
                    *clipped_cpu_columns = k;
                    clipped_cpu_values++;
                    clipped_cpu_columns++;
                    break;
                }
            }
        }
    }
    return clipped;
}

template <typename Dtype>
ConvDictWrapper<Dtype>::ConvDictWrapper(cusparseHandle_t *handle, const Blob<Dtype> *Dl, int N,
    CSCParameter::Boundary boundary, Dtype lambda2)
    : handle_(handle), solver_handle_(NULL), n_(Dl->shape(0)), m_(Dl->shape(1)), N_(N),
    boundary_(boundary), lambda2_(lambda2),
    D_(new CSRWrapper<Dtype>(handle, N_, N_*m_, n_*m_*N_)),
    DtDpl2I_(new CSRWrapper<Dtype>(handle, N_*m_, N_*m_, -1)) {
    CHECK_EQ(boundary_, CSCParameter::CIRCULANT_BACK)
        << "Only circulant back boundary condtion is currently supported!";
    make_conv_dict_gpu(n_, m_, Dl->gpu_data(), N_, boundary_,
        D_->mutable_values(), D_->mutable_columns(), D_->mutable_ptrB());
    CUSOLVER_CHECK(cusolverSpCreate(&solver_handle_));
}

template <typename Dtype>
ConvDictWrapper<Dtype>::~ConvDictWrapper() {
    CUSOLVER_CHECK(cusolverSpDestroy(solver_handle_));
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
        int nnz_local;
        CUDA_CHECK(cudaMemcpy(&nnz_local, DtDpl2I_->ptrB()+DtDpl2I_->row(), sizeof(int),
            cudaMemcpyDeviceToHost));
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

template <>
void ConvDictWrapper<float>::solve(int nnz, const int *h_inds, float *d_x) {
    CHECK_GE(DtDpl2I_->nnz(), 0); // should be initialized by create()
    shared_ptr<CSRWrapper<float> > clipped = DtDpl2I_->clip(nnz, h_inds);
    float tol = 1e-6;
    int reorder = 0;
    int singularity = 0;
    CUSOLVER_CHECK(cusolverSpScsrlsvchol(solver_handle_, clipped->row(), clipped->nnz(),
        clipped->descr(), clipped->values(), clipped->ptrB(), clipped->columns(),
        d_x, tol, reorder, d_x, &singularity));
    CHECK_EQ(singularity, -1) << "Singularity value is " << singularity;
}

template <>
void ConvDictWrapper<double>::solve(int nnz, const int *h_inds, double *d_x) {
    CHECK_GE(DtDpl2I_->nnz(), 0); // should be initialized by create()
    shared_ptr<CSRWrapper<double> > clipped = DtDpl2I_->clip(nnz, h_inds);
    double tol = 1e-9;
    int reorder = 0;
    int singularity = 0;
    CUSOLVER_CHECK(cusolverSpDcsrlsvchol(solver_handle_, clipped->row(), clipped->nnz(),
        clipped->descr(), clipped->values(), clipped->ptrB(), clipped->columns(),
        d_x, tol, reorder, d_x, &singularity));
    CHECK_EQ(singularity, -1) << "Singularity value is " << singularity;
}

template <typename Dtype>
void ConvDictWrapper<Dtype>::analyse(int nnz, const int *h_inds, const Dtype *d_x) {
    shared_ptr<CSRWrapper<Dtype> > clipped = DtDpl2I_->clip(nnz, h_inds);
    int issym = 0;
    CUSOLVER_CHECK(cusolverSpXcsrissymHost(solver_handle_, clipped->row(), clipped->nnz(),
        clipped->descr(), clipped->cpu_ptrB(), clipped->cpu_ptrE(), clipped->cpu_columns(), &issym));
    LOG_IF(INFO, issym == 1) << "The clipped matrix is symmetric.";
    LOG_IF(WARNING, issym == 0) << "The clipped matrix is NOT symmetric.";
}

INSTANTIATE_CLASS(CSRWrapper);
INSTANTIATE_CLASS(ConvDictWrapper);

} // namespace caffe
