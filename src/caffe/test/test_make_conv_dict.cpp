#include <vector>

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
}

} // namespace caffe
