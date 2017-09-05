#include "caffe/util/conv_dict_wrapper.hpp"

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
// template <typename Dtype>
// ConvDictWrapper<Dtype>::ConvDictWrapper(const Blob<Dtype> *Dl,
//     int N, CSCParameter::Boundary boundary, Dtype lambda2)
//     : boundary_(boundary), lambda2_(lambda2) {
//     CHECK_EQ(boundary_, CSCParameter::CIRCULANT_BACK) <<
//         "Only circulant boundary is supported!";
//     int n = Dl->shape(0);
//     int m = Dl->shape(1);
//     h_values_ = malloc(sizeof(Dtype)*n*m*N);
//     h_columns_ = malloc(sizeof(int)*n*m*N);
//     h_ptrB_ = malloc(sizeof(int)*(m*N+1));
//     h_ptrE_ = h_ptrB_ + 1;
//     n_ = n;
//     m_ = m;
// }
//
// template <typename Dtype>
// ConvDictWrapper<Dtype>::~ConvDictWrapper() {
//     free(h_values_);
//     free(h_columns_);
//     free(h_ptrB_);
// }
//
// template <typename Dtype>
// void ConvDictWrapper<Dtype>::make_conv_dict() {
//
// }

} // namespace caffe
