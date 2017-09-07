#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/conv_dict_wrapper.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

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
    }
}

TYPED_TEST(CSRWrapperTest, TestSetColumns) {
    const int columns[] = {0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4};
    int h_columns[13];
    this->csr_wrapper_->set_columns(columns);
    CUDA_CHECK(cudaMemcpy(h_columns, this->csr_wrapper_->columns(), 13*sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 13; ++i) {
        EXPECT_EQ(h_columns[i], columns[i]);
    }
}

TYPED_TEST(CSRWrapperTest, TestSetPtrB) {
    const int ptrB[] = {0, 3, 5, 8, 11, 13};
    int h_ptrB[6];
    this->csr_wrapper_->set_ptrB(ptrB);
    CUDA_CHECK(cudaMemcpy(h_ptrB, this->csr_wrapper_->ptrB(), 6*sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(h_ptrB[i], ptrB[i]);
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

} // namespace caffe
