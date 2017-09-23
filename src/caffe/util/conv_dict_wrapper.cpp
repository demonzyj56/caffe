#include "caffe/util/conv_dict_wrapper.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {


template <typename Dtype>
void make_conv_dict_cpu(const int n, const int m, const Dtype *Dl, const int N,
    CSCParameter::Boundary boundary, Dtype *values, int *columns, int *ptrB) {
    LOG(WARNING) << "This version is deprecated.";
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

/*
 * Creating the circulant dictionary from the local one.
 * The local dictionary has size of n x m, and the circulant dictionary has size
 * (channels x height x width) x (height x width x m).
 * For circulant case, each row/column of D has the same number of elements,
 * We use a simplified method, namely to create the CSC format of D.  In this case,
 * ptrB has a length of cxhxw.  However, please note that Dl is still row-major.
 * nnz = nxNxm.
 * Note also that the create sparse matrix is unsorted.  On GPU end should call
 * csrsort to sort the indices.
 * */
template <typename Dtype>
void make_transposed_conv_dict_circulant_cpu(int n, int m, const Dtype *Dl, int channels, int height, int width,
        int kernel_h, int kernel_w, int pad_h, int pad_w, Dtype *values, int *columns,
        int *ptrB) {
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            caffe_copy(n*m, Dl, values);
            // values += n*m;
            for (int mm = 0; mm < m; ++mm) {
                for (int c = 0; c < channels; ++c) {
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        int h_offset = (h + kh - pad_h + height) % height;
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int w_offset = (w + kw - pad_w + width) % width;
                            int Dl_row = (c * kernel_h + kh) * kernel_w + kw;
                            *values++ = Dl[Dl_row * m + mm];
                            *columns++ = (c * height + h_offset) * width + w_offset;
                            // values++;
                            // columns++;
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i <= m * height * width; ++i) {
        ptrB[i] = i * n;
    }
}

template <typename Dtype>
void make_transposed_conv_dict_padzeros_cpu(int n, int m, const Dtype *Dl, int channels, int height, int width,
        int kernel_h, int kernel_w, int pad_h, int pad_w, Dtype *values, int *columns, int *ptrB) {
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int mm = 0; mm < m; ++mm) {
                for (int c = 0; c < channels; ++c) {
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        int h_offset = h + kh - pad_h;
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int w_offset = w + kw - pad_w;
                            int Dl_row = (c * kernel_h + kh) * kernel_w + kw;
                            *values++ = Dl[Dl_row * m + mm];
                            if (h_offset >= 0 && h_offset < height && w_offset >= 0 && w_offset < width) {
                                *columns++ = (c * height + h_offset) * width + w_offset;
                            } else {
                                *columns++ = -1;
                            }
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i <= m * height * width; ++i) {
        ptrB[i] = i * n;
    }
}

// dispatch
// Since we don't know exactly the nnz in values/columns, we create the memory of size
// channels*kernel_h*kernel_w*m*height*width, where the invalid entries in columns is -1.
template <typename Dtype>
void make_transposed_conv_dict_cpu(int n, int m, const Dtype *Dl, int channels, int height, int width,
        int kernel_h, int kernel_w, CSCParameter::Boundary boundary, Dtype *values, int *columns,
        int *ptrB) {
    CHECK_EQ(n, channels * kernel_h * kernel_w);
    switch(boundary) {
        case CSCParameter::CIRCULANT_BACK:
            make_transposed_conv_dict_circulant_cpu(n, m, Dl, channels, height, width, kernel_h, kernel_w, 
                0, 0, values, columns, ptrB);
            break;
        case CSCParameter::CIRCULANT_FRONT:
            make_transposed_conv_dict_circulant_cpu(n, m, Dl, channels, height, width, kernel_h, kernel_w, 
                kernel_h-1, kernel_w-1, values, columns, ptrB);
            break;
        case CSCParameter::PAD_BACK:
            make_transposed_conv_dict_padzeros_cpu(n, m, Dl, channels, height, width, kernel_h, kernel_w, 
                0, 0, values, columns, ptrB);
            break;
        case CSCParameter::PAD_FRONT:
            make_transposed_conv_dict_padzeros_cpu(n, m, Dl, channels, height, width, kernel_h, kernel_w, 
                kernel_h-1, kernel_w-1, values, columns, ptrB);
            break;
        case CSCParameter::PAD_BOTH:
            CHECK_EQ(kernel_h % 2, 1);
            CHECK_EQ(kernel_w % 2, 1);
            make_transposed_conv_dict_padzeros_cpu(n, m, Dl, channels, height, width, kernel_h, kernel_w,
                (kernel_h-1)/2, (kernel_w-1)/2, values, columns, ptrB);
            break;
        case CSCParameter::NOPAD:
            LOG(FATAL) << "Non padding boundary condition is not supported!";
        default:
            NOT_IMPLEMENTED;
    }
}

template void make_conv_dict_cpu<float>(const int n, const int m, const float *Dl, const int N,
    CSCParameter::Boundary boundary, float *values, int *columns, int *ptrB);
template void make_conv_dict_cpu<double>(const int n, const int m, const double *Dl, const int N,
    CSCParameter::Boundary boundary, double *values, int *columns, int *ptrB);

template void make_transposed_conv_dict_cpu<float>(int n, int m, const float *Dl, int channels, int height,
    int width, int kernel_h, int kernel_w, CSCParameter::Boundary boundary, float *values, int *columns,
    int *ptrB);
template void make_transposed_conv_dict_cpu<double>(int n, int m, const double *Dl, int channels, int height,
    int width, int kernel_h, int kernel_w, CSCParameter::Boundary boundary, double *values, int *columns,
    int *ptrB);

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
    CHECK(nnz_ >= 0) << "nnz is " << nnz_;
    if (NULL == d_values_) {
        d_values_ = shared_ptr<SyncedMemory>(new SyncedMemory(sizeof(Dtype)*nnz_));
    }
    return (Dtype *)d_values_->mutable_gpu_data();
}

template <typename Dtype>
int *CSRWrapper<Dtype>::mutable_columns() {
    CHECK(nnz_ >= 0) << "nnz is " << nnz_;
    if (NULL == d_columns_) {
        d_columns_ = shared_ptr<SyncedMemory>(new SyncedMemory(sizeof(int)*nnz_));
    }
    return (int *)d_columns_->mutable_gpu_data();
}

template <typename Dtype>
int *CSRWrapper<Dtype>::mutable_ptrB() {
    CHECK(r_ > 0) << "row number is " << r_;
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

// Before clipping, check whether h_inds have duplicate entries.
// If so then die.
template <typename Dtype>
shared_ptr<CSRWrapper<Dtype> > CSRWrapper<Dtype>::clip(int nnz, const int *h_inds) {
    LOG(WARNING) << "This version is deprecated.";
    for (int i = 0; i < nnz-1; ++i) {
        CHECK_LT(h_inds[i], h_inds[i+1]) << "The index set should be strictly increasing.";
    }
    shared_ptr<CSRWrapper<Dtype> > clipped(new CSRWrapper<Dtype>(handle_, nnz, nnz, -1));
    clipped->mutable_cpu_ptrB()[0] = 0;
    for (int i = 0; i < nnz; ++i) {
        int ind = h_inds[i];
        clipped->mutable_cpu_ptrB()[i+1] = clipped->mutable_cpu_ptrB()[i];
        for (int j = cpu_ptrB()[ind]; j < cpu_ptrE()[ind]; ++j) {
            int target = cpu_columns()[j];
            for (int k = 0; k < nnz; ++k) {
                if (h_inds[k] == target) {
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
        int ind = h_inds[i];
        for (int j = cpu_ptrB()[ind]; j < cpu_ptrE()[ind]; ++j) {
            Dtype target_value = cpu_values()[j];
            int target_column = cpu_columns()[j];
            for (int k = 0; k < nnz; ++k) {
                if (h_inds[k] == target_column) {
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
shared_ptr<CSRWrapper<Dtype> > CSRWrapper<Dtype>::clip_columns(int nnz, const int *h_inds) {
    LOG(WARNING) << "This version is deprecated.";
    for (int i = 0; i < nnz-1; ++i) {
        CHECK_LT(h_inds[i], h_inds[i+1]) << "The index set should be strictly increasing.";
    }
    shared_ptr<CSRWrapper<Dtype> > clipped(new CSRWrapper<Dtype>(handle_, r_, nnz, -1));
    clipped->mutable_cpu_ptrB()[0] = 0;
    for (int i = 0; i < r_; ++i) {
        clipped->mutable_cpu_ptrB()[i+1] = clipped->mutable_cpu_ptrB()[i];
        for (int j = cpu_ptrB()[i]; j < cpu_ptrE()[i]; ++j) {
            int target = cpu_columns()[j];
            for (int k = 0; k < nnz; ++k) {
                if (h_inds[k] == target) {
                    clipped->mutable_cpu_ptrB()[i+1] += 1;
                    break;
                }
            }
        }
    }
    clipped->set_nnz(clipped->cpu_ptrB()[r_]);
    Dtype *clipped_cpu_values = clipped->mutable_cpu_values();
    int *clipped_cpu_columns = clipped->mutable_cpu_columns();
    for (int i = 0; i < r_; ++i) {
        for (int j = cpu_ptrB()[i]; j < cpu_ptrE()[i]; ++j) {
            Dtype target_value = cpu_values()[j];
            int target_column = cpu_columns()[j];
            for (int k = 0; k < nnz; ++k) {
                if (h_inds[k] == target_column) {
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

// should create a cusolver handle
template <typename Dtype>
bool CSRWrapper<Dtype>::symmetric() {
    if (r_ != c_) {
        LOG(WARNING) << "Not a square matrix.";
        return false;
    }
    cusolverSpHandle_t handle = NULL;
    int issym = 0;
    CUSOLVER_CHECK(cusolverSpCreate(&handle));
    CUSOLVER_CHECK(cusolverSpXcsrissymHost(handle, r_, nnz_, descr_,
        cpu_ptrB(), cpu_ptrE(), cpu_columns(), &issym));
    CUSOLVER_CHECK(cusolverSpDestroy(handle));
    return static_cast<bool>(issym);
}

template <>
void CSRWrapper<float>::sort() {
    size_t pBufferSizeInBytes = 0;
    CUSPARSE_CHECK(cusparseXcsrsort_bufferSizeExt(*handle_, row(), col(), nnz(), ptrB(),
        columns(), &pBufferSizeInBytes));
    SyncedMemory pBuffer(pBufferSizeInBytes);
    SyncedMemory P(sizeof(int)*nnz());
    SyncedMemory valBuffer(sizeof(float)*nnz());
    CUSPARSE_CHECK(cusparseCreateIdentityPermutation(*handle_, nnz(),
        (int *)P.mutable_gpu_data()));
    CUSPARSE_CHECK(cusparseXcsrsort(*handle_, row(), col(), nnz(), descr(), ptrB(), mutable_columns(),
        (int *)P.mutable_gpu_data(), pBuffer.mutable_gpu_data()));
    CUSPARSE_CHECK(cusparseSgthr(*handle_, nnz(), mutable_values(),
        (float *)valBuffer.mutable_gpu_data(), (const int *)P.gpu_data(),
        cusparseGetMatIndexBase(descr())));
    caffe_copy(nnz(), (const float *)valBuffer.gpu_data(), mutable_values());
}

template <>
void CSRWrapper<double>::sort() {
    size_t pBufferSizeInBytes = 0;
    CUSPARSE_CHECK(cusparseXcsrsort_bufferSizeExt(*handle_, row(), col(), nnz(), ptrB(),
        columns(), &pBufferSizeInBytes));
    SyncedMemory pBuffer(pBufferSizeInBytes);
    SyncedMemory P(sizeof(int)*nnz());
    SyncedMemory valBuffer(sizeof(double)*nnz());
    CUSPARSE_CHECK(cusparseCreateIdentityPermutation(*handle_, nnz(),
        (int *)P.mutable_gpu_data()));
    CUSPARSE_CHECK(cusparseXcsrsort(*handle_, row(), col(), nnz(), descr(), ptrB(), mutable_columns(),
        (int *)P.mutable_gpu_data(), pBuffer.mutable_gpu_data()));
    CUSPARSE_CHECK(cusparseDgthr(*handle_, nnz(), mutable_values(),
        (double *)valBuffer.mutable_gpu_data(), (const int *)P.gpu_data(),
        cusparseGetMatIndexBase(descr())));
    caffe_copy(nnz(), (const double *)valBuffer.gpu_data(), mutable_values());
}

template <typename Dtype>
ConvDictWrapper<Dtype>::ConvDictWrapper(cusparseHandle_t *handle, const Blob<Dtype> *Dl, int N,
    CSCParameter::Boundary boundary, Dtype lambda2)
    : handle_(handle), dnsolver_handle_(NULL), spsolver_handle_(NULL), n_(Dl->shape(0)),
    m_(Dl->shape(1)), N_(N), boundary_(boundary), lambda2_(lambda2),
    D_(new CSRWrapper<Dtype>(handle, N_, N_*m_, n_*m_*N_)),
    DtDpl2I_(), debug_(false) {
    LOG(WARNING) << "This version is deprecated.";
    CHECK_EQ(boundary_, CSCParameter::CIRCULANT_BACK)
        << "Only circulant back boundary condtion is currently supported!";
    make_conv_dict_gpu(n_, m_, Dl->gpu_data(), N_, boundary_,
        D_->mutable_values(), D_->mutable_columns(), D_->mutable_ptrB());
    CUSOLVER_CHECK(cusolverSpCreate(&spsolver_handle_));
    CUSOLVER_CHECK(cusolverDnCreate(&dnsolver_handle_));
}

template <typename Dtype>
ConvDictWrapper<Dtype>::ConvDictWrapper(cusparseHandle_t *handle, const Blob<Dtype> *Dl, int channels,
        int height, int width, int kernel_h, int kernel_w, CSCParameter::Boundary boundary,
        Dtype lambda2)
        : handle_(handle), dnsolver_handle_(NULL), spsolver_handle_(NULL), n_(Dl->shape(0)),
        m_(Dl->shape(1)), N_(-1), boundary_(boundary), lambda2_(lambda2),
        D_(), DtDpl2I_(), debug_(false), channels_(channels), height_(height), width_(width),
        kernel_h_(kernel_h), kernel_w_(kernel_w) {
    CHECK_EQ(n_, channels_ * kernel_h_ * kernel_w_);
    CUSOLVER_CHECK(cusolverSpCreate(&spsolver_handle_));
    CUSOLVER_CHECK(cusolverDnCreate(&dnsolver_handle_));
    this->make_conv_dict(Dl);
    CHECK(D_);
}

template <typename Dtype>
ConvDictWrapper<Dtype>::~ConvDictWrapper() {
    CUSOLVER_CHECK(cusolverSpDestroy(spsolver_handle_));
    CUSOLVER_CHECK(cusolverDnDestroy(dnsolver_handle_));
}

// Split into three steps:
// 1. create the transposed and unpruned version of D_.
// 2. prune (in place).
// 3. transpose (out of place).
template <typename Dtype>
void ConvDictWrapper<Dtype>::make_conv_dict(const Blob<Dtype> *Dl) {
    CHECK_EQ(Dl->shape(0), n_);
    CHECK_EQ(Dl->shape(1), m_);
    CSRWrapper<Dtype> Dtrans(handle_, m_*height_*width_, channels_*height_*width_, n_*m_*height_*width_);
    make_transposed_conv_dict_gpu(n_, m_, Dl->gpu_data(), channels_, height_, width_, kernel_h_, kernel_w_,
        boundary_, Dtrans.mutable_values(), Dtrans.mutable_columns(), Dtrans.mutable_ptrB());
    Dtrans.prune();
    D_ = Dtrans.transpose();
}

template <>
void ConvDictWrapper<float>::create() {
    LOG(WARNING) << "This version is deprecated.";
    if (NULL == DtDpl2I_) {
        DtDpl2I_.reset(new CSRWrapper<float>(handle_, m_*height_*width_,
                m_*height_*width_, -1));
    }
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
    CUDA_CHECK(cudaMemcpy(&nnz, DtDpl2I_->ptrB()+DtDpl2I_->row(), sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&base, DtDpl2I_->ptrB(), sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_EQ(base, 0);
    nnz -= base;
    if (NULL != nnzTotalDevHostPtr) {
        CHECK_EQ(*nnzTotalDevHostPtr, nnz) << "Different values of nonzero!";
    }
    LOG_IF(INFO, debug_) << "Nonzeros: " << nnz
        << ", sparsity: " << float(nnz)/DtDpl2I_->row()/DtDpl2I_->col();
    CHECK_GE(nnz, 0) << "An overflow of int32 value is suspected. "
        << "Sadly there is noting we could do.  Dying...";
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
    LOG(WARNING) << "This version is deprecated.";
    if (NULL == DtDpl2I_) {
        DtDpl2I_.reset(new CSRWrapper<double>(handle_, m_*height_*width_,
                m_*height_*width_, -1));
    }
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
shared_ptr<CSRWrapper<float> > ConvDictWrapper<float>::create_clipped(int nind, const int *h_ind) {
    // step 0: create transposed D and identity matrix and output
    shared_ptr<CSRWrapper<float> > Dclipped = D_->clip_columns_gpu(nind, h_ind);
    shared_ptr<CSRWrapper<float> > Dtrans = Dclipped->transpose();
    shared_ptr<CSRWrapper<float> > identity =
        shared_ptr<CSRWrapper<float> >(new CSRWrapper<float>(
            handle_, Dclipped->col(), Dclipped->col(), Dclipped->col()));
    shared_ptr<CSRWrapper<float> > lhs = shared_ptr<CSRWrapper<float> >(
        new CSRWrapper<float>(handle_, Dclipped->col(), Dclipped->col(), -1));
    identity->identity();
    int base, nnz;
    csrgemm2Info_t info = NULL;
    size_t bufferSize;
    int *nnzTotalDevHostPtr = &nnz;
    float alpha = 1.;
    float beta = lambda2_;
    CUSPARSE_CHECK(cusparseSetPointerMode(*handle_, CUSPARSE_POINTER_MODE_HOST));
    // step 1: create an opaque structure
    CUSPARSE_CHECK(cusparseCreateCsrgemm2Info(&info));
    // step 2: allocate buffer for csrgemm2Nnz and csrgemm2
    CUSPARSE_CHECK(cusparseScsrgemm2_bufferSizeExt(
        *handle_, Dclipped->col(), Dclipped->col(), Dclipped->row(), &alpha,
        Dtrans->descr(), Dtrans->nnz(), Dtrans->ptrB(), Dtrans->columns(),
        Dclipped->descr(), Dclipped->nnz(), Dclipped->ptrB(), Dclipped->columns(),
        &beta,
        identity->descr(), identity->nnz(), identity->ptrB(), identity->columns(),
        info, &bufferSize));
    SyncedMemory buffer(bufferSize);
    //step 3: compute csrRowPtrC
    CUSPARSE_CHECK(cusparseXcsrgemm2Nnz(*handle_, Dclipped->col(), Dclipped->col(), Dclipped->row(),
        Dtrans->descr(), Dtrans->nnz(), Dtrans->ptrB(), Dtrans->columns(),
        Dclipped->descr(), Dclipped->nnz(), Dclipped->ptrB(), Dclipped->columns(),
        identity->descr(), identity->nnz(), identity->ptrB(), identity->columns(),
        lhs->descr(), lhs->mutable_ptrB(), nnzTotalDevHostPtr,
        info, buffer.mutable_gpu_data()));
    CUDA_CHECK(cudaMemcpy(&nnz, lhs->ptrB()+lhs->row(), sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&base, lhs->ptrB(), sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_EQ(base, 0);
    nnz -= base;
    if (NULL != nnzTotalDevHostPtr) {
        CHECK_EQ(*nnzTotalDevHostPtr, nnz) << "Different values of nonzero!";
    }
    LOG_IF(INFO, debug_) << "LHS matrix nonzeros: " << nnz
        << ", sparsity: " << float(nnz)/lhs->row()/lhs->col();
    CHECK_GE(nnz, 0) << "An overflow of int32 value is suspected. "
        << "Sadly there is noting we could do.  Dying...";
    lhs->set_nnz(nnz);
    // step 4: finith sparsity pattern and value of C
    CUSPARSE_CHECK(cusparseScsrgemm2(*handle_, Dclipped->col(), Dclipped->col(), Dclipped->row(), &alpha,
        Dtrans->descr(), Dtrans->nnz(), Dtrans->values(), Dtrans->ptrB(), Dtrans->columns(),
        Dclipped->descr(), Dclipped->nnz(), Dclipped->values(), Dclipped->ptrB(), Dclipped->columns(),
        &beta,
        identity->descr(), identity->nnz(), identity->values(), identity->ptrB(), identity->columns(),
        lhs->descr(), lhs->mutable_values(), lhs->mutable_ptrB(), lhs->mutable_columns(),
        info, buffer.mutable_gpu_data()));
    // step 5: destroy and free memory
    CUSPARSE_CHECK(cusparseDestroyCsrgemm2Info(info));
    return lhs;
}

template <>
shared_ptr<CSRWrapper<double> > ConvDictWrapper<double>::create_clipped(int nind, const int *h_ind) {
    // step 0: create transposed D and identity matrix and output
    shared_ptr<CSRWrapper<double> > Dclipped = D_->clip_columns_gpu(nind, h_ind);
    shared_ptr<CSRWrapper<double> > Dtrans = Dclipped->transpose();
    shared_ptr<CSRWrapper<double> > identity =
        shared_ptr<CSRWrapper<double> >(new CSRWrapper<double>(
            handle_, Dclipped->col(), Dclipped->col(), Dclipped->col()));
    shared_ptr<CSRWrapper<double> > lhs = shared_ptr<CSRWrapper<double> >(
        new CSRWrapper<double>(handle_, Dclipped->col(), Dclipped->col(), -1));
    identity->identity();
    int base, nnz;
    csrgemm2Info_t info = NULL;
    size_t bufferSize;
    int *nnzTotalDevHostPtr = &nnz;
    double alpha = 1.;
    double beta = lambda2_;
    CUSPARSE_CHECK(cusparseSetPointerMode(*handle_, CUSPARSE_POINTER_MODE_HOST));
    // step 1: create an opaque structure
    CUSPARSE_CHECK(cusparseCreateCsrgemm2Info(&info));
    // step 2: allocate buffer for csrgemm2Nnz and csrgemm2
    CUSPARSE_CHECK(cusparseDcsrgemm2_bufferSizeExt(
        *handle_, Dclipped->col(), Dclipped->col(), Dclipped->row(), &alpha,
        Dtrans->descr(), Dtrans->nnz(), Dtrans->ptrB(), Dtrans->columns(),
        Dclipped->descr(), Dclipped->nnz(), Dclipped->ptrB(), Dclipped->columns(),
        &beta,
        identity->descr(), identity->nnz(), identity->ptrB(), identity->columns(),
        info, &bufferSize));
    SyncedMemory buffer(bufferSize);
    //step 3: compute csrRowPtrC
    CUSPARSE_CHECK(cusparseXcsrgemm2Nnz(*handle_, Dclipped->col(), Dclipped->col(), Dclipped->row(),
        Dtrans->descr(), Dtrans->nnz(), Dtrans->ptrB(), Dtrans->columns(),
        Dclipped->descr(), Dclipped->nnz(), Dclipped->ptrB(), Dclipped->columns(),
        identity->descr(), identity->nnz(), identity->ptrB(), identity->columns(),
        lhs->descr(), lhs->mutable_ptrB(), nnzTotalDevHostPtr,
        info, buffer.mutable_gpu_data()));
    CUDA_CHECK(cudaMemcpy(&nnz, lhs->ptrB()+lhs->row(), sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&base, lhs->ptrB(), sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_EQ(base, 0);
    nnz -= base;
    if (NULL != nnzTotalDevHostPtr) {
        CHECK_EQ(*nnzTotalDevHostPtr, nnz) << "Different values of nonzero!";
    }
    LOG_IF(INFO, debug_) << "LHS matrix nonzeros: " << nnz
        << ", sparsity: " << double(nnz)/lhs->row()/lhs->col();
    CHECK_GE(nnz, 0) << "An overflow of int32 value is suspected. "
        << "Sadly there is noting we could do.  Dying...";
    lhs->set_nnz(nnz);
    // step 4: finith sparsity pattern and value of C
    CUSPARSE_CHECK(cusparseDcsrgemm2(*handle_, Dclipped->col(), Dclipped->col(), Dclipped->row(), &alpha,
        Dtrans->descr(), Dtrans->nnz(), Dtrans->values(), Dtrans->ptrB(), Dtrans->columns(),
        Dclipped->descr(), Dclipped->nnz(), Dclipped->values(), Dclipped->ptrB(), Dclipped->columns(),
        &beta,
        identity->descr(), identity->nnz(), identity->values(), identity->ptrB(), identity->columns(),
        lhs->descr(), lhs->mutable_values(), lhs->mutable_ptrB(), lhs->mutable_columns(),
        info, buffer.mutable_gpu_data()));
    // step 5: destroy and free memory
    CUSPARSE_CHECK(cusparseDestroyCsrgemm2Info(info));
    return lhs;
}

template <>
void ConvDictWrapper<float>::solve(int nnz, const int *h_inds, float *d_x) {
    CPUTimer timer;
    timer.Start();
    shared_ptr<CSRWrapper<float> > clipped;
    if (NULL != DtDpl2I_) {
        clipped = DtDpl2I_->clip(nnz, h_inds);
        LOG_IF(INFO, debug_) << "Using precomputed matrix to solve.";
    } else {
        clipped = create_clipped(nnz, h_inds);
    }
    LOG_IF(INFO, debug_) << "Clip time: " << timer.Seconds() << "s.";
    timer.Start();
    float tol = 1e-12;
    int reorder = 0;
    int singularity = 0;
    CUSOLVER_CHECK(cusolverSpScsrlsvchol(spsolver_handle_, clipped->row(), clipped->nnz(),
        clipped->descr(), clipped->values(), clipped->ptrB(), clipped->columns(),
        d_x, tol, reorder, d_x, &singularity));
    LOG_IF(INFO, debug_) << "Actual solving time: " << timer.Seconds() << "s.";
    CHECK_EQ(singularity, -1) << "Singularity value is " << singularity;
}

template <>
void ConvDictWrapper<double>::solve(int nnz, const int *h_inds, double *d_x) {
    CPUTimer timer;
    timer.Start();
    shared_ptr<CSRWrapper<double> > clipped;
    if (NULL != DtDpl2I_) {
        clipped = DtDpl2I_->clip(nnz, h_inds);
        LOG_IF(INFO, debug_) << "Using precomputed matrix to solve.";
    } else {
        clipped = create_clipped(nnz, h_inds);
    }
    LOG_IF(INFO, debug_) << "Clip time: " << timer.Seconds() << "s.";
    timer.Start();
    double tol = 1e-12;
    int reorder = 0;
    int singularity = 0;
    CUSOLVER_CHECK(cusolverSpDcsrlsvchol(spsolver_handle_, clipped->row(), clipped->nnz(),
        clipped->descr(), clipped->values(), clipped->ptrB(), clipped->columns(),
        d_x, tol, reorder, d_x, &singularity));
    LOG_IF(INFO, debug_) << "Actual solving time: " << timer.Seconds() << "s.";
    CHECK_EQ(singularity, -1) << "Singularity value is " << singularity;
}

template <>
void ConvDictWrapper<float>::analyse(int nnz, const int *h_inds) {
    shared_ptr<CSRWrapper<float> > clipped;
    if (NULL != DtDpl2I_) {
        clipped = DtDpl2I_->clip(nnz, h_inds);
        LOG(INFO) << "Analysing precomputed matrix.";
    } else {
        clipped = create_clipped(nnz, h_inds);
    }
    CHECK(clipped->symmetric()) << "The clipped matrix is NOT symmetric.";
    SyncedMemory clipped_dense(clipped->row()*clipped->col()*sizeof(float));
    float *clipped_dense_gpu = (float *)clipped_dense.mutable_gpu_data();
    clipped->to_dense(clipped_dense_gpu);
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR; // only eigenvalues are needed
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
    SyncedMemory eigs(clipped->row()*sizeof(float));
    SyncedMemory devInfo(sizeof(int));
    int lwork = 0;
    // use dense routine to check positive-definitiveness
    CUSOLVER_CHECK(cusolverDnSsyevd_bufferSize(dnsolver_handle_, jobz, uplo, clipped->row(),
        clipped_dense_gpu, clipped->row(), (const float *)eigs.gpu_data(), &lwork));
    SyncedMemory work(lwork*sizeof(float));
    CUSOLVER_CHECK(cusolverDnSsyevd(dnsolver_handle_, jobz, uplo, clipped->row(),
        clipped_dense_gpu, clipped->row(), (float *)eigs.mutable_gpu_data(),
        (float *)work.mutable_gpu_data(), lwork, (int *)devInfo.mutable_gpu_data()));
    CHECK_EQ(*(const int *)devInfo.cpu_data(), 0);
    for (int i = 0; i < clipped->row(); ++i) {
        float eig = ((const float *)eigs.cpu_data())[i];
        CHECK_GE(eig, lambda2_);
        LOG(INFO) << "The " << i+1 << "th eigenvalue is " << eig;
    }
}

template <>
void ConvDictWrapper<double>::analyse(int nnz, const int *h_inds) {
    shared_ptr<CSRWrapper<double> > clipped;
    if (NULL != DtDpl2I_) {
        clipped = DtDpl2I_->clip(nnz, h_inds);
        LOG(INFO) << "Analysing precomputed matrix.";
    } else {
        clipped = create_clipped(nnz, h_inds);
    }
    CHECK(clipped->symmetric()) << "The clipped matrix is NOT symmetric.";
    SyncedMemory clipped_dense(clipped->row()*clipped->col()*sizeof(double));
    double *clipped_dense_gpu = (double *)clipped_dense.mutable_gpu_data();
    clipped->to_dense(clipped_dense_gpu);
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR; // only eigenvalues are needed
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
    SyncedMemory eigs(clipped->row()*sizeof(double));
    SyncedMemory devInfo(sizeof(int));
    int lwork = 0;
    // use dense routine to check positive-definitiveness
    CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(dnsolver_handle_, jobz, uplo, clipped->row(),
        clipped_dense_gpu, clipped->row(), (const double *)eigs.gpu_data(), &lwork));
    SyncedMemory work(lwork*sizeof(double));
    CUSOLVER_CHECK(cusolverDnDsyevd(dnsolver_handle_, jobz, uplo, clipped->row(),
        clipped_dense_gpu, clipped->row(), (double *)eigs.mutable_gpu_data(),
        (double *)work.mutable_gpu_data(), lwork, (int *)devInfo.mutable_gpu_data()));
    CHECK_EQ(*(const int *)devInfo.cpu_data(), 0);
    for (int i = 0; i < clipped->row(); ++i) {
        double eig = ((const double *)eigs.cpu_data())[i];
        CHECK_GE(eig, lambda2_);
        LOG(INFO) << "The " << i+1 << "th eigenvalue is " << eig;
    }
}

INSTANTIATE_CLASS(CSRWrapper);
INSTANTIATE_CLASS(ConvDictWrapper);

} // namespace caffe
