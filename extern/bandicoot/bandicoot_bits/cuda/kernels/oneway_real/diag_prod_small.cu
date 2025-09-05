// Copyright 2023 Ryan Curtin (http://www.ratml.org/)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

// compute the product of the diagonal of a matrix
__global__
void
COOT_FN(PREFIX,diag_prod_small)(const eT1* in_mem,
                                const UWORD n_rows,
                                eT1* out_mem)
  {
  eT1* aux_mem = (eT1*) aux_shared_mem;

  const UWORD tid = threadIdx.x;
  UWORD i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  const UWORD grid_size = blockDim.x * 2 * gridDim.x;

  aux_mem[tid] = 1;

  while (i + blockDim.x < n_rows)
    {
    // copy to local shared memory
    const UWORD index1 = i * n_rows + i;
    const eT1 v1 = in_mem[index1];
    const UWORD index2 = (i + blockDim.x) * n_rows + (i + blockDim.x);
    const eT1 v2 = in_mem[index2];
    aux_mem[tid] *= v1 * v2;
    i += grid_size;
    }
  if (i < n_rows)
    {
    const UWORD index = i * n_rows + i;
    const eT1 v = in_mem[index];
    aux_mem[tid] *= v;
    }

  for (UWORD s = blockDim.x / 2; s > 0; s >>= 1)
    {
    if (tid < s)
      {
      aux_mem[tid] *= aux_mem[tid + s];
      }
    }

  if (tid == 0)
    {
    out_mem[blockIdx.x] = aux_mem[0];
    }
  }
