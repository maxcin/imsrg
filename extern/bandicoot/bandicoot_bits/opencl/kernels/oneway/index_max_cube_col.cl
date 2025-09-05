// Copyright 2025 Ryan Curtin (http://www.ratml.org)
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

__kernel
void
COOT_FN(PREFIX,index_max_cube_col)(__global UWORD* dest,
                                   const UWORD dest_offset,
                                   __global const eT1* src,
                                   const UWORD src_offset,
                                   const UWORD n_rows,
                                   const UWORD n_cols,
                                   const UWORD n_slices)
  {
  const UWORD row = get_global_id(0);
  const UWORD slice = get_global_id(1);

  if(row < n_rows && slice < n_slices)
    {
    eT1 best_val = src[src_offset + row + slice * n_rows * n_cols];
    UWORD best_index = 0;
    #pragma unroll
    for(UWORD i=1; i < n_cols; ++i)
      {
      if (src[src_offset + (i * n_rows) + row + slice * n_rows * n_cols] > best_val)
        {
        best_val = src[src_offset + (i * n_rows) + row + slice * n_rows * n_cols];
        best_index = i;
        }
      }
    dest[dest_offset + row + slice * n_rows] = best_index;
    }
  }
