#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);
  __m256 mvec = _mm256_load_ps(m);
  __m256 zero = _mm256_setzero_ps();
  float jidx[N] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
  for(int i=0; i<N; i++) {
    /*for(int j=0; j<N; j++) {
      if(i != j) {
        float rx = x[i] - x[j];
        float ry = y[i] - y[j];
        float r = std::sqrt(rx * rx + ry * ry);
        fx[i] -= rx * m[j] / (r * r * r);
        fy[i] -= ry * m[j] / (r * r * r);
      }
    }*/
    __m256 jvec = _mm256_load_ps(jidx);
    __m256 ivec = _mm256_set1_ps(i);
    __m256 mask = _mm256_cmp_ps(jvec, ivec, _CMP_NEQ_OQ);
     
    __m256 xivec = _mm256_set1_ps(x[i]);
    __m256 yivec = _mm256_set1_ps(y[i]);

    __m256 fxvec = _mm256_load_ps(fx);
    __m256 fyvec = _mm256_load_ps(fy);

    __m256 rxvec = _mm256_sub_ps(xivec, xvec);
    __m256 ryvec = _mm256_sub_ps(yivec, yvec);
    __m256 rxpvec = _mm256_mul_ps(rxvec, rxvec);
    __m256 rypvec = _mm256_mul_ps(ryvec, ryvec);
    __m256 rxysvec = _mm256_add_ps(rxpvec, rypvec);
    __m256 rrvec = _mm256_rsqrt_ps(rxysvec);

    fxvec = _mm256_mul_ps(rxvec, mvec);
    fxvec = _mm256_mul_ps(fxvec, rrvec);
    fxvec = _mm256_mul_ps(fxvec, rrvec);
    fxvec = _mm256_mul_ps(fxvec, rrvec);
    fxvec = _mm256_blendv_ps(zero, fxvec, mask);
    fyvec = _mm256_mul_ps(ryvec, mvec);
    fyvec = _mm256_mul_ps(fyvec, rrvec);
    fyvec = _mm256_mul_ps(fyvec, rrvec);
    fyvec = _mm256_mul_ps(fyvec, rrvec);
    fyvec = _mm256_blendv_ps(zero, fyvec, mask);
    fxvec = _mm256_sub_ps(zero, fxvec);
    fyvec = _mm256_sub_ps(zero, fyvec);
    __m256 fsumvec = _mm256_permute2f128_ps(fxvec, fxvec, 1);
    fsumvec = _mm256_add_ps(fsumvec, fxvec);
    fsumvec = _mm256_hadd_ps(fsumvec, fsumvec);
    fsumvec = _mm256_hadd_ps(fsumvec, fsumvec);    
    _mm256_store_ps(fx, fsumvec);
    fsumvec = _mm256_permute2f128_ps(fyvec, fyvec, 1);
    fsumvec = _mm256_add_ps(fsumvec, fyvec);
    fsumvec = _mm256_hadd_ps(fsumvec, fsumvec);
    fsumvec = _mm256_hadd_ps(fsumvec, fsumvec);    
    _mm256_store_ps(fy, fsumvec);
 
    printf("%d %g %g\n",i,fx[i],fy[i]);
    /*for(int i = 0; i<N; i++){
      printf("%g\n", fx[i]);
    }*/
  }
}
