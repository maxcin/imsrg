// Copyright 2022 Ryan Curtin (http://www.ratml.org/)
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
COOT_FN(PREFIX,htrans)(__global eT2* out,
                       const UWORD out_offset,
                       __global const eT1* in,
                       const UWORD in_offset,
                       const UWORD in_n_rows,
                       const UWORD in_n_cols)
  {
  // For a non-inplace transpose, we can use a pretty naive approach.
  const UWORD row = get_global_id(0);
  const UWORD col = get_global_id(1);
  const UWORD in_total_offset = in_offset + row + col * in_n_rows;
  const UWORD out_total_offset = out_offset + col + row * in_n_cols;

  if( (row < in_n_rows) && (col < in_n_cols) )
    {
    const eT2 element = (eT2) in[in_total_offset];
    out[out_total_offset] = COOT_FN(conj_,eT2)(element);
    }
  }
