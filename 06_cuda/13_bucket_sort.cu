#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void bucket_sort(int *key, int *bucket) {
  int t_i = threadIdx.x;
  atomicAdd(&bucket[key[t_i]], 1); 
  __syncthreads();
  int value = 0;
  while(t_i >=0) {
    t_i -= bucket[value++];
  }
  key[threadIdx.x] = value-1;
  __syncthreads();
}

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");
/*
  std::vector<int> bucket(range); 
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
  for (int i=0; i<n; i++) {
    bucket[key[i]]++;
  }
  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }
*/

  int *key_g;
  cudaMallocManaged(&key_g, n*sizeof(int));
  cudaMemcpy(key_g, &key[0], sizeof(int)*n, cudaMemcpyHostToDevice);
  int *bucket;
  cudaMallocManaged(&bucket, range*sizeof(int));

  bucket_sort<<<1, n>>>(key_g, bucket);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    key[i] = key_g[i];
  }

  cudaFree(key_g);
  cudaFree(bucket);

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
