#include <vector>
#include <algorithm>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/conv_dict_wrapper.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class MakeConvDictTest : public GPUDeviceTest<Dtype> {

protected:
    MakeConvDictTest() {
        n_ = 7;
        m_ = 3;
        N_ = 20;
        boundary_ = CSCParameter::CIRCULANT_BACK;
        vector<int> Dl_shape(2);
        Dl_shape[0] = n_;
        Dl_shape[1] = m_;
        Dl_ = new Blob<Dtype>(Dl_shape);
        h_values_ = new Dtype[n_*m_*N_];
        h_values_copy_ = new Dtype[n_*m_*N_];
        h_columns_ = new int[n_*m_*N_];
        h_columns_copy_ = new int[n_*m_*N_];
        h_ptrB_ = new int[N_+1];
        h_ptrB_copy_ = new int[N_+1];
        CUDA_CHECK(cudaMalloc((void **)&d_values_, sizeof(Dtype)*n_*m_*N_));
        CUDA_CHECK(cudaMalloc((void **)&d_columns_, sizeof(int)*n_*m_*N_));
        CUDA_CHECK(cudaMalloc((void **)&d_ptrB_, sizeof(int)*(N_+1)));
    }
    
    virtual ~MakeConvDictTest() {
        delete Dl_;
        delete [] h_values_;
        delete [] h_values_copy_;
        delete [] h_columns_;
        delete [] h_columns_copy_;
        delete [] h_ptrB_;
        delete [] h_ptrB_copy_;
        CUDA_CHECK(cudaFree(d_values_));
        CUDA_CHECK(cudaFree(d_columns_));
        CUDA_CHECK(cudaFree(d_ptrB_));
    }

    virtual void SetUp() {
        FillerParameter filler_param;
        GaussianFiller<Dtype> filler(filler_param);
        filler.Fill(this->Dl_);
    }

    int n_;
    int m_;
    int N_;
    CSCParameter::Boundary boundary_;
    Blob<Dtype>* Dl_;
    Dtype* h_values_;
    Dtype* h_values_copy_;
    Dtype* d_values_;
    int* h_columns_;
    int* h_columns_copy_;
    int* d_columns_;
    int* h_ptrB_;
    int* h_ptrB_copy_;
    int* d_ptrB_;
};


TYPED_TEST_CASE(MakeConvDictTest, TestDtypes);

TYPED_TEST(MakeConvDictTest, TestNothing) {
}

TYPED_TEST(MakeConvDictTest, TestCPURoutineSanity) {
    const TypeParam *Dl_cpu_ptr = this->Dl_->cpu_data();
    make_conv_dict_cpu(this->n_, this->m_, Dl_cpu_ptr, this->N_, this->boundary_,
        this->h_values_, this->h_columns_, this->h_ptrB_);
}

TYPED_TEST(MakeConvDictTest, TestGPURoutineSanity) {
    const TypeParam *Dl_gpu_ptr = this->Dl_->gpu_data();
    make_conv_dict_gpu(this->n_, this->m_, Dl_gpu_ptr, this->N_, this->boundary_,
        this->d_values_, this->d_columns_, this->d_ptrB_);
}

TYPED_TEST(MakeConvDictTest, TestResultCoincide) {
    const TypeParam *Dl_cpu_ptr = this->Dl_->cpu_data();
    const TypeParam *Dl_gpu_ptr = this->Dl_->gpu_data();
    make_conv_dict_cpu(this->n_, this->m_, Dl_cpu_ptr, this->N_, this->boundary_,
        this->h_values_, this->h_columns_, this->h_ptrB_);
    make_conv_dict_gpu(this->n_, this->m_, Dl_gpu_ptr, this->N_, this->boundary_,
        this->d_values_, this->d_columns_, this->d_ptrB_);
    CUDA_CHECK(cudaMemcpy(this->h_values_copy_, this->d_values_,
        sizeof(TypeParam)*this->n_*this->m_*this->N_, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(this->h_columns_copy_, this->d_columns_,
        sizeof(int)*this->n_*this->m_*this->N_, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(this->h_ptrB_copy_, this->d_ptrB_,
        sizeof(int)*(this->N_+1), cudaMemcpyDeviceToHost));
    for (int i = 0; i < this->n_*this->m_*this->N_; ++i) {
        EXPECT_EQ(this->h_values_[i], this->h_values_copy_[i]);
        EXPECT_EQ(this->h_columns_[i], this->h_columns_copy_[i]);
    }
    for (int i = 0; i < this->N_+1; ++i) {
        EXPECT_EQ(this->h_ptrB_[i], this->h_ptrB_copy_[i]);
    }
    EXPECT_EQ(this->h_ptrB_[this->N_], this->n_*this->m_*this->N_);
}

template <typename Dtype>
class CusparseHandleTest : public GPUDeviceTest<Dtype> {
protected:
    CusparseHandleTest() : handle_wrapper_() {
    }

    virtual ~CusparseHandleTest() {
    }

    CusparseHandle handle_wrapper_;
};

TYPED_TEST_CASE(CusparseHandleTest, TestDtypes);

TYPED_TEST(CusparseHandleTest, TestHandleSanity) {
    EXPECT_NE(*this->handle_wrapper_.get(), (cusparseHandle_t)NULL);
}

template <typename Dtype>
class CSRWrapperTest : public GPUDeviceTest<Dtype> {
protected:
    CSRWrapperTest() : handle_() {
        r_ = 5;
        c_ = 5;
        nnz_ = 13;
        csr_wrapper_ = new CSRWrapper<Dtype>(handle_.get(), r_, c_, nnz_);
    }

    virtual ~CSRWrapperTest() {
        delete csr_wrapper_;
    }

    int r_;
    int c_;
    int nnz_;
    CusparseHandle handle_;
    CSRWrapper<Dtype> * csr_wrapper_;
};

TYPED_TEST_CASE(CSRWrapperTest, TestDtypes);

TYPED_TEST(CSRWrapperTest, TestCreation) {
    EXPECT_EQ(this->csr_wrapper_->row(), this->r_);
    EXPECT_EQ(this->csr_wrapper_->col(), this->c_);
    EXPECT_EQ(this->csr_wrapper_->nnz(), this->nnz_);
}

TYPED_TEST(CSRWrapperTest, TestSetNnz) {
    this->csr_wrapper_->set_nnz(10);
    EXPECT_EQ(this->csr_wrapper_->nnz(), 10);
}

TYPED_TEST(CSRWrapperTest, TestSetValues) {
    const TypeParam values[] = {1., -1., -3., -2., 5., 4., 6., 4., -4., 2., 7., 8., -5.};
    TypeParam h_values[13];
    this->csr_wrapper_->set_values(values);
    CUDA_CHECK(cudaMemcpy(h_values, this->csr_wrapper_->values(), 13*sizeof(TypeParam),
        cudaMemcpyDeviceToHost));
    for (int i = 0; i < 13; ++i) {
        EXPECT_EQ(h_values[i], values[i]);
        EXPECT_EQ(this->csr_wrapper_->cpu_values()[i], values[i]);
    }
}

TYPED_TEST(CSRWrapperTest, TestSetColumns) {
    const int columns[] = {0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4};
    int h_columns[13];
    this->csr_wrapper_->set_columns(columns);
    CUDA_CHECK(cudaMemcpy(h_columns, this->csr_wrapper_->columns(), 13*sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 13; ++i) {
        EXPECT_EQ(h_columns[i], columns[i]);
        EXPECT_EQ(this->csr_wrapper_->cpu_columns()[i], columns[i]);
    }
}

TYPED_TEST(CSRWrapperTest, TestSetPtrB) {
    const int ptrB[] = {0, 3, 5, 8, 11, 13};
    int h_ptrB[6];
    this->csr_wrapper_->set_ptrB(ptrB);
    CUDA_CHECK(cudaMemcpy(h_ptrB, this->csr_wrapper_->ptrB(), 6*sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(h_ptrB[i], ptrB[i]);
        EXPECT_EQ(this->csr_wrapper_->cpu_ptrB()[i], ptrB[i]);
    }
}

TYPED_TEST(CSRWrapperTest, TestSetMatrixType) {
    const cusparseMatrixType_t matrix_types[] = {
        CUSPARSE_MATRIX_TYPE_GENERAL,
        CUSPARSE_MATRIX_TYPE_SYMMETRIC,
        CUSPARSE_MATRIX_TYPE_HERMITIAN,
        CUSPARSE_MATRIX_TYPE_TRIANGULAR
    };
    for (int i = 0; i < 4; ++i) {
        this->csr_wrapper_->set_matrix_type(matrix_types[i]);
        EXPECT_EQ(cusparseGetMatType(this->csr_wrapper_->descr()), matrix_types[i]);
    }
}

TYPED_TEST(CSRWrapperTest, TestSetDiagType) {
    const cusparseDiagType_t diag_types[] = {
        CUSPARSE_DIAG_TYPE_NON_UNIT,
        CUSPARSE_DIAG_TYPE_UNIT
    };
    for (int i = 0; i < 2; ++i) {
        this->csr_wrapper_->set_diag_type(diag_types[i]);
        EXPECT_EQ(cusparseGetMatDiagType(this->csr_wrapper_->descr()), diag_types[i]);
    }
}

TYPED_TEST(CSRWrapperTest, TestSetFillMode) {
    const cusparseFillMode_t fill_modes[] = {
        CUSPARSE_FILL_MODE_LOWER,
        CUSPARSE_FILL_MODE_UPPER
    };
    for (int i = 0; i < 2; ++i) {
        this->csr_wrapper_->set_fill_mode(fill_modes[i]);
        EXPECT_EQ(cusparseGetMatFillMode(this->csr_wrapper_->descr()), fill_modes[i]);
    }
}

TYPED_TEST(CSRWrapperTest, TestSetIndexBase) {
    const cusparseIndexBase_t index_bases[] = {
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_INDEX_BASE_ONE
    };
    for (int i = 0; i < 2; ++i) {
        this->csr_wrapper_->set_index_base(index_bases[i]);
        EXPECT_EQ(cusparseGetMatIndexBase(this->csr_wrapper_->descr()), index_bases[i]);
    }
}

TYPED_TEST(CSRWrapperTest, TestTranspose) {
    const TypeParam values[] = {1., -1., -3., -2., 5., 4., 6., 4., -4., 2., 7., 8., -5.};
    const TypeParam values_t[] = {1., -2., -4., -1., 5., 8., 4., 2., -3., 6., 7., 4., -5.};
    const int columns[] = {0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4};
    const int columns_t[] = {0, 1, 3, 0, 1, 4, 2, 3, 0, 2, 3, 2, 4};
    const int ptrB[] = {0, 3, 5, 8, 11, 13};
    const int ptrB_t[] = {0, 3, 6, 8, 11, 13};
    TypeParam h_values_t[13];
    int h_columns_t[13];
    int h_ptrB_t[6];
    this->csr_wrapper_->set_values(values);
    this->csr_wrapper_->set_columns(columns);
    this->csr_wrapper_->set_ptrB(ptrB);
    shared_ptr<CSRWrapper<TypeParam> > mat_trans = this->csr_wrapper_->transpose();
    CUDA_CHECK(cudaMemcpy(h_values_t, mat_trans->values(), 13*sizeof(TypeParam),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_columns_t, mat_trans->columns(), 13*sizeof(int), 
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ptrB_t, mat_trans->ptrB(), 6*sizeof(int),
        cudaMemcpyDeviceToHost));
    for (int i = 0; i < 13; ++i) {
        EXPECT_EQ(values_t[i], h_values_t[i]);
        EXPECT_EQ(columns_t[i], h_columns_t[i]);
    }
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(ptrB_t[i], h_ptrB_t[i]);
    }
}

TYPED_TEST(CSRWrapperTest, TestIdentity) {
    this->csr_wrapper_->set_nnz(5);
    this->csr_wrapper_->identity();
    TypeParam h_values[5];
    int h_columns[5];
    int h_ptrB[5];
    CUDA_CHECK(cudaMemcpy(h_values, this->csr_wrapper_->values(), 5*sizeof(TypeParam),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_columns, this->csr_wrapper_->columns(), 5*sizeof(int),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ptrB, this->csr_wrapper_->ptrB(), 6*sizeof(int),
        cudaMemcpyDeviceToHost));
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(h_values[i], TypeParam(1));
        EXPECT_EQ(h_columns[i], i);
    }
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(h_ptrB[i], i);
    }
}

TYPED_TEST(CSRWrapperTest, TestToDense) {
    const TypeParam values[] = {1., -1., -3., -2., 5., 4., 6., 4., -4., 2., 7., 8., -5.};
    const int columns[] = {0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4};
    const int ptrB[] = {0, 3, 5, 8, 11, 13};
    this->csr_wrapper_->set_values(values);
    this->csr_wrapper_->set_columns(columns);
    this->csr_wrapper_->set_ptrB(ptrB);
    SyncedMemory dense(this->r_*this->c_*sizeof(TypeParam));
    this->csr_wrapper_->to_dense((TypeParam *)dense.mutable_gpu_data());
    const TypeParam *cpu_ptr = (const TypeParam *)dense.cpu_data();
    for (int r = 0; r < this->r_; ++r) {
        for (int c = ptrB[r]; c < ptrB[r+1]; ++c) {
            int column = columns[c];
            EXPECT_EQ(values[c], cpu_ptr[r*this->c_+column]);
        }
    }
}

TYPED_TEST(CSRWrapperTest, TestClip) {
    const TypeParam values[] = {1., -1., -3., -2., 5., 4., 6., 4., -4., 2., 7., 8., -5.};
    const int columns[] = {0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4};
    const int ptrB[] = {0, 3, 5, 8, 11, 13};
    const TypeParam values_clipped[] = {1., -1., -2., 5., 8., -5.};
    const int columns_clipped[] = {0, 1, 0, 1, 1, 2};
    const int ptrB_clipped[] = {0, 2, 4, 6};
    const int inds[] = {0, 1, 4};
    this->csr_wrapper_->set_values(values);
    this->csr_wrapper_->set_columns(columns);
    this->csr_wrapper_->set_ptrB(ptrB);
    shared_ptr<CSRWrapper<TypeParam> > clipped = this->csr_wrapper_->clip(3, inds);
    ASSERT_EQ(clipped->nnz(), 6);  // should die if wrong before proceed
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(clipped->cpu_values()[i], values_clipped[i]);
        EXPECT_EQ(clipped->cpu_columns()[i], columns_clipped[i]);
    }
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(clipped->cpu_ptrB()[i], ptrB_clipped[i]);
    }
}

template <typename Dtype>
class CSRWrapperTransposeTest : public GPUDeviceTest<Dtype> {
protected:
    CSRWrapperTransposeTest()
        : handle_(), csr_wrapper_(new CSRWrapper<Dtype>(handle_.get(), 3, 5, 8)) {
        const Dtype values[] = {1., -1., -3., -2., 5., 4., 6., 4.};
        const int columns[] = {0, 1, 3, 0, 1, 2, 3, 4};
        const int ptrB[] = {0, 3, 5, 8};
        csr_wrapper_->set_values(values);
        csr_wrapper_->set_columns(columns);
        csr_wrapper_->set_ptrB(ptrB);
        csr_trans_ = csr_wrapper_->transpose();
    }
    virtual ~CSRWrapperTransposeTest() {
        delete csr_wrapper_;
    }

    CusparseHandle handle_;
    CSRWrapper<Dtype> * const csr_wrapper_;
    shared_ptr<CSRWrapper<Dtype> > csr_trans_;
};

TYPED_TEST_CASE(CSRWrapperTransposeTest, TestDtypes);

TYPED_TEST(CSRWrapperTransposeTest, TestTransposeShape) {
    EXPECT_EQ(this->csr_wrapper_->row(), this->csr_trans_->col());
    EXPECT_EQ(this->csr_wrapper_->col(), this->csr_trans_->row());
    EXPECT_EQ(this->csr_wrapper_->nnz(), this->csr_trans_->nnz());
}

TYPED_TEST(CSRWrapperTransposeTest, TestTransposeValues) {
    const TypeParam h_values_t[] = {1., -2., -1., 5., 4., -3., 6., 4.};
    TypeParam h_values[8];
    CUDA_CHECK(cudaMemcpy(h_values, this->csr_trans_->values(), 8*sizeof(TypeParam),
        cudaMemcpyDeviceToHost));
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(h_values_t[i], h_values[i]);
    }
}

TYPED_TEST(CSRWrapperTransposeTest, TestTransposeColumns) {
    const int h_columns_t[] = {0, 1, 0, 1, 2, 0, 2, 2};
    int h_columns[8];
    CUDA_CHECK(cudaMemcpy(h_columns, this->csr_trans_->columns(), 8*sizeof(int),
        cudaMemcpyDeviceToHost));
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(h_columns_t[i], h_columns[i]);
    }
}

TYPED_TEST(CSRWrapperTransposeTest, TestTransposePtrB) {
    const int h_ptrB_t[] = {0, 2, 4, 5, 7, 8};
    int h_ptrB[6];
    CUDA_CHECK(cudaMemcpy(h_ptrB, this->csr_trans_->ptrB(), 6*sizeof(int),
        cudaMemcpyDeviceToHost));
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(h_ptrB_t[i], h_ptrB[i]);
    }
}

TYPED_TEST(CSRWrapperTransposeTest, TestTransposeTranspose) {
    shared_ptr<CSRWrapper<TypeParam> > csr_tt = this->csr_trans_->transpose();
    TypeParam h_values[8];
    TypeParam h_values_tt[8];
    CUDA_CHECK(cudaMemcpy(h_values, this->csr_wrapper_->values(), 8*sizeof(TypeParam),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_values_tt, csr_tt->values(), 8*sizeof(TypeParam),
        cudaMemcpyDeviceToHost));
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(h_values[i], h_values_tt[i]);
    }
}


template <typename Dtype>
class ConvDictWrapperTest : public GPUDeviceTest<Dtype> {
protected:
    ConvDictWrapperTest()
    : handle_(), conv_dict_(NULL), Dl_(NULL), N_(20),
    boundary_(CSCParameter::CIRCULANT_BACK), lambda2_(0.1) {
        vector<int> Dl_shape(2);
        Dl_shape[0] = 7;
        Dl_shape[1] = 3;
        Dl_ = new Blob<Dtype>(Dl_shape);
        FillerParameter filler_param;
        filler_param.set_min(0.5);
        filler_param.set_max(1);
        UniformFiller<Dtype> filler(filler_param);
        filler.Fill(Dl_);
        conv_dict_ = new ConvDictWrapper<Dtype>(handle_.get(),
            Dl_, N_, boundary_, lambda2_);
    }
    virtual ~ConvDictWrapperTest() {
        delete Dl_;
        delete conv_dict_;
    }

    CusparseHandle handle_;
    ConvDictWrapper<Dtype> *conv_dict_;
    Blob<Dtype> *Dl_;
    int N_;
    CSCParameter::Boundary boundary_;
    Dtype lambda2_;
};

TYPED_TEST_CASE(ConvDictWrapperTest, TestDtypes);

TYPED_TEST(ConvDictWrapperTest, TestSetUp) {
}

TYPED_TEST(ConvDictWrapperTest, TestConvDictSize) {
    EXPECT_EQ(this->conv_dict_->D()->row(), this->N_);
    EXPECT_EQ(this->conv_dict_->D()->col(), this->N_*this->Dl_->shape(1));
    if (this->boundary_ == CSCParameter::CIRCULANT_BACK ||
        this->boundary_ == CSCParameter::CIRCULANT_FRONT) {
        EXPECT_EQ(this->conv_dict_->D()->nnz(), this->N_*this->Dl_->count());
    } else {
        NOT_IMPLEMENTED;
    }
}

TYPED_TEST(ConvDictWrapperTest, TestConvDictSanity) {
    SyncedMemory dense(this->Dl_->shape(1)*this->N_*this->N_*sizeof(TypeParam));
    this->conv_dict_->D()->to_dense((TypeParam *)dense.mutable_gpu_data());
    const TypeParam *cpu_ptr = (const TypeParam *)dense.cpu_data();
    for (int b = 0; b < this->N_; ++b) {
        for (int r = 0; r < this->Dl_->shape(0); ++r) {
            for (int c = 0; c < this->Dl_->shape(1); ++c) {
                int row = (b + r) % this->N_;
                int col = b * this->Dl_->shape(1) + c;
                int ind = row * (this->N_*this->Dl_->shape(1)) + col;
                EXPECT_EQ(cpu_ptr[ind], this->Dl_->cpu_data()[r*this->Dl_->shape(1)+c])
                    << " Block: " << b
                    << " r: " << r
                    << " c: " << c
                    << " row: " << row
                    << " col: " << col
                    << " ind: " << ind;
            }
        }
    }
}


TYPED_TEST(ConvDictWrapperTest, TestLhsMatCreation) {
    int n = this->Dl_->shape(0);
    int m = this->Dl_->shape(1);
    this->conv_dict_->create();
    EXPECT_EQ(this->conv_dict_->DtDpl2I()->row(), this->N_*m);
    EXPECT_EQ(this->conv_dict_->DtDpl2I()->col(), this->N_*m);
    EXPECT_TRUE(this->conv_dict_->DtDpl2I()->symmetric());
    if (this->boundary_ == CSCParameter::CIRCULANT_BACK ||
        this->boundary_ == CSCParameter::CIRCULANT_FRONT) {
        EXPECT_EQ(this->conv_dict_->DtDpl2I()->nnz(), this->N_*(2*n-1)*m*m);
    } else {
        NOT_IMPLEMENTED;
    }
}

TYPED_TEST(ConvDictWrapperTest, TestSolve) {
    int nnz = 10;
    vector<int> inds(this->conv_dict_->DtDpl2I()->row());
    for (int i = 0; i < inds.size(); ++i) {
        inds[i] = i;
    }
    std::random_shuffle(inds.begin(), inds.end());
    std::sort(inds.begin(), inds.begin() + nnz);
    SyncedMemory rhs(sizeof(TypeParam)*nnz);
    TypeParam *rhs_ptr = (TypeParam *)rhs.mutable_gpu_data();
    caffe_gpu_rng_gaussian(nnz, TypeParam(0), TypeParam(1), rhs_ptr);
    this->conv_dict_->create();
    this->conv_dict_->solve(nnz, inds.data(), rhs_ptr);
    this->conv_dict_->analyse(nnz, inds.data());
}

} // namespace caffe
