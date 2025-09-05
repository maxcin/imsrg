// Copyright 2023-2025 Ryan Curtin (http://www.ratml.org/)
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

// this kernel is technically incorrect if the size is not a factor of 2!
__global__
void
COOT_FN(PREFIX,approx_equal_cube)(uint* out_mem,
                                  const eT1* A_mem,
                                  const UWORD A_M_n_rows,
                                  const UWORD A_M_n_cols,
                                  const eT1* B_mem,
                                  const UWORD B_M_n_rows,
                                  const UWORD B_M_n_cols,
                                  const UWORD n_rows,
                                  const UWORD n_cols,
                                  const UWORD n_elem,
                                  const UWORD mode,
                                  const eT1 abs_tol,
                                  const eT1 rel_tol)
  {
  uint* aux_mem = (uint*) aux_shared_mem;

  const UWORD tid = threadIdx.x;
  UWORD i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  const UWORD grid_size = blockDim.x * 2 * gridDim.x;

  const UWORD n_elem_slice = n_rows * n_cols;

  aux_mem[tid] = 1;

  while (i + blockDim.x < n_elem)
    {
    // A bit painful... TODO: implement a more efficient non-modulo approach
    const UWORD elem1 = i % n_elem_slice;
    const UWORD slice1 = i / n_elem_slice;
    const UWORD row1 = elem1 % n_rows;
    const UWORD col1 = elem1 / n_rows;

    const UWORD elem2 = (i + blockDim.x) % n_elem_slice;
    const UWORD slice2 = (i + blockDim.x) / n_elem_slice;
    const UWORD row2 = elem2 % n_rows;
    const UWORD col2 = elem2 / n_rows;

    const UWORD A_loc1 = row1 + col1 * A_M_n_rows + slice1 * A_M_n_rows * A_M_n_cols;
    const UWORD A_loc2 = row2 + col2 * A_M_n_rows + slice2 * A_M_n_rows * A_M_n_cols;
    const UWORD B_loc1 = row1 + col1 * B_M_n_rows + slice1 * B_M_n_rows * B_M_n_cols;
    const UWORD B_loc2 = row2 + col2 * B_M_n_rows + slice2 * B_M_n_rows * B_M_n_cols;

    const eT1 A_val1 = A_mem[A_loc1];
    const eT1 B_val1 = B_mem[B_loc1];
    const eT1 A_val2 = A_mem[A_loc2];
    const eT1 B_val2 = B_mem[B_loc2];

    if (coot_isnan(A_val1) || coot_isnan(B_val1) || coot_isnan(A_val2) || coot_isnan(B_val2))
      {
      // Not approximately equal.
      aux_mem[tid] &= 0;
      }

    const eT1 absdiff1 = COOT_FN(PREFIX,absdiff)(A_val1, B_val1);
    const eT1 absdiff2 = COOT_FN(PREFIX,absdiff)(A_val2, B_val2);

    if ((mode & 1) == 1) // absolute
      {
      aux_mem[tid] &= (absdiff1 <= abs_tol);
      aux_mem[tid] &= (absdiff2 <= abs_tol);
      }

    if ((mode & 2) == 2) // relative
      {
      const eT1 max_val1 = max(ET1_ABS(A_val1), ET1_ABS(B_val1));
      const eT1 max_val2 = max(ET1_ABS(A_val2), ET1_ABS(B_val2));

      if (max_val1 >= (eT1) 1)
        {
        aux_mem[tid] &= (absdiff1 <= rel_tol * max_val1);
        aux_mem[tid] &= (absdiff2 <= rel_tol * max_val2);
        }
      else
        {
        aux_mem[tid] &= (absdiff1 / max_val1 <= rel_tol);
        aux_mem[tid] &= (absdiff2 / max_val2 <= rel_tol);
        }
      }

    i += grid_size;
    }
  if (i < n_elem)
    {
    const UWORD elem = i % n_elem_slice;
    const UWORD slice = i / n_elem_slice;
    const UWORD row = elem % n_rows;
    const UWORD col = elem / n_rows;

    const UWORD A_loc = row + col * A_M_n_rows + slice * A_M_n_rows * A_M_n_cols;
    const UWORD B_loc = row + col * B_M_n_rows + slice * B_M_n_rows * B_M_n_cols;

    const eT1 A_val = A_mem[A_loc];
    const eT1 B_val = B_mem[B_loc];

    if (coot_isnan(A_val) || coot_isnan(B_val))
      {
      // Not approximately equal.
      aux_mem[tid] &= 0;
      }

    const eT1 absdiff = COOT_FN(PREFIX,absdiff)(A_val, B_val);

    if ((mode & 1) == 1) // absolute
      {
      aux_mem[tid] &= (absdiff <= abs_tol);
      }

    if ((mode & 2) == 2) // relative
      {
      const eT1 max_val = max(ET1_ABS(A_val), ET1_ABS(B_val));

      if (max_val >= (eT1) 1)
        {
        aux_mem[tid] &= (absdiff <= rel_tol * max_val);
        }
      else
        {
        aux_mem[tid] &= (absdiff / max_val <= rel_tol);
        }
      }
    }
  __syncthreads();

  for (UWORD s = blockDim.x / 2; s > 32; s >>= 1)
    {
    if (tid < s)
      {
      aux_mem[tid] &= aux_mem[tid + s];
      }
    __syncthreads();
  }

  if (tid < 32) // unroll last warp's worth of work
    {
    u32_and_warp_reduce(aux_mem, tid);
    }

  if (tid == 0)
    {
    out_mem[blockIdx.x] = aux_mem[0];
    }
  }
