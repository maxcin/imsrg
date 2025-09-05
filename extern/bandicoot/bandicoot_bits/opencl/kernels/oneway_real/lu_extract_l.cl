// Copyright 2023 Ryan Curtin (http://www.ratml.org)
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

// This extracts L and U from in, and sets the lower diagonal of U to 0.
// It's okay if U == in, if n_rows <= n_cols.
__kernel
void
COOT_FN(PREFIX,lu_extract_l)(__global eT1* L,
                             const UWORD L_offset,
                             __global eT1* U,
                             const UWORD U_offset,
                             const __global eT1* in,
                             const UWORD in_offset,
                             const UWORD n_rows,
                             const UWORD n_cols)
  {
  const UWORD row = get_global_id(0);
  const UWORD col = get_global_id(1);

  // Note that neither U nor L must be square.
  //   L has size n_rows x min(n_rows, n_cols).
  //   U has size min(n_rows, n_cols) x n_cols.
  const UWORD min_rows_cols = min(n_rows, n_cols);

  const UWORD in_index = row + n_rows * col; // this is also L_out_index
  const UWORD U_out_index = row + min_rows_cols * col;

  if ((row < n_rows) && (col < min_rows_cols))
    {
    L[L_offset + in_index] = (row > col) ? in[in_offset + in_index] : ((row == col) ? 1 : 0);
    }

  if ((row < min_rows_cols) && (col < n_cols))
    {
    U[U_offset + U_out_index] = (row > col) ? 0 : in[in_offset + in_index];
    }
  }
