// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023 Ryan Curtin (http://www.ratml.org)
// Copyright 2022 Gopi Tatiraju
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



template<typename out_eT, typename T1, typename T2>
inline
void
glue_join_rows::apply(Mat<out_eT>& out, const Glue<T1, T2, glue_join_rows>& glue)
  {
  coot_extra_debug_sigprint();

  const std::string func_name = (glue.aux_uword == 0) ? "join_rows()" : "join_horiz()";

  const no_conv_unwrap<T1> U1(glue.A);
  const no_conv_unwrap<T2> U2(glue.B);

  // check for same number of columns
  const uword A_n_rows = U1.M.n_rows;
  const uword A_n_cols = U1.M.n_cols;

  const uword B_n_rows = U2.M.n_rows;
  const uword B_n_cols = U2.M.n_cols;

  coot_debug_check
    (
    ( (A_n_rows != B_n_rows) && ( (A_n_rows > 0) || (A_n_cols > 0) ) && ( (B_n_rows > 0) || (B_n_cols > 0) ) ),
    func_name + ": number of rows must be the same"
    );

  const uword new_n_rows = (std::max)(A_n_rows, B_n_rows);
  const uword new_n_cols = A_n_cols + B_n_cols;

  // Shortcut: if there is nothing to do, leave early.
  if (new_n_rows == 0 || new_n_cols == 0)
    {
    out.set_size(new_n_rows, new_n_cols);
    return;
    }

  // We can't have the output be an alias of the input.
  typedef typename no_conv_unwrap<T1>::stored_type UT1;
  typedef typename no_conv_unwrap<T2>::stored_type UT2;
  alias_wrapper<Mat<out_eT>, UT1, UT2> W(out, U1.M, U2.M);
  W.use.set_size(new_n_rows, new_n_cols);
  coot_rt_t::join_rows(W.get_dev_mem(false),
                       U1.get_dev_mem(false), A_n_rows, A_n_cols,
                       U2.get_dev_mem(false), B_n_rows, B_n_cols,
                       U1.get_dev_mem(false), 0, 0, /* ignored */
                       U1.get_dev_mem(false), 0, 0, /* ignored */
                       // subview arguments
                       W.get_row_offset(), W.get_col_offset(), W.get_M_n_rows(),
                       U1.get_row_offset(), U1.get_col_offset(), U1.get_M_n_rows(),
                       U2.get_row_offset(), U2.get_col_offset(), U2.get_M_n_rows(),
                       0, 0, 1, /* ignored */
                       0, 0, 1 /* ignored */);
  }



template<typename eT, typename T1, typename T2, typename T3>
inline
void
glue_join_rows::apply(Mat<eT>& out, const T1& A, const T2& B, const T3& C, const std::string& func_name)
  {
  coot_extra_debug_sigprint();

  const no_conv_unwrap<T1> U1(A);
  const no_conv_unwrap<T2> U2(B);
  const no_conv_unwrap<T3> U3(C);

  // check for same number of columns
  const uword A_n_rows = U1.M.n_rows;
  const uword A_n_cols = U1.M.n_cols;
  const uword B_n_rows = U2.M.n_rows;
  const uword B_n_cols = U2.M.n_cols;
  const uword C_n_rows = U3.M.n_rows;
  const uword C_n_cols = U3.M.n_cols;

  const uword out_n_cols = A_n_cols + B_n_cols + C_n_cols;
  const uword out_n_rows = ((std::max)((std::max)(A_n_rows, B_n_rows), C_n_rows));

  coot_debug_check( ((A_n_rows != out_n_rows) && ((A_n_rows > 0) || (A_n_cols > 0))), func_name + ": number of rows must be the same" );
  coot_debug_check( ((B_n_rows != out_n_rows) && ((B_n_rows > 0) || (B_n_cols > 0))), func_name + ": number of rows must be the same" );
  coot_debug_check( ((C_n_rows != out_n_rows) && ((C_n_rows > 0) || (C_n_cols > 0))), func_name + ": number of rows must be the same" );

  // Shortcut: if there is nothing to do, leave early.
  if (out_n_rows == 0 || out_n_cols == 0)
    {
    out.set_size(out_n_rows, out_n_cols);
    return;
    }

  // We can't have the output be an alias of the input.
  typedef typename no_conv_unwrap<T1>::stored_type UT1;
  typedef typename no_conv_unwrap<T2>::stored_type UT2;
  typedef typename no_conv_unwrap<T3>::stored_type UT3;
  alias_wrapper<Mat<eT>, UT1, UT2, UT3> W(out, U1.M, U2.M, U3.M);
  W.use.set_size(out_n_rows, out_n_cols);
  coot_rt_t::join_rows(W.get_dev_mem(false),
                       U1.get_dev_mem(false), A_n_rows, A_n_cols,
                       U2.get_dev_mem(false), B_n_rows, B_n_cols,
                       U3.get_dev_mem(false), C_n_rows, C_n_cols,
                       U1.get_dev_mem(false), 0, 0, /* ignored */
                       // subview arguments
                       W.get_row_offset(), W.get_col_offset(), W.get_M_n_rows(),
                       U1.get_row_offset(), U1.get_col_offset(), U1.get_M_n_rows(),
                       U2.get_row_offset(), U2.get_col_offset(), U2.get_M_n_rows(),
                       U3.get_row_offset(), U3.get_col_offset(), U3.get_M_n_rows(),
                       0, 0, 1 /* ignored */);
  }



template<typename eT, typename T1, typename T2, typename T3, typename T4>
inline
void
glue_join_rows::apply(Mat<eT>& out, const T1& A, const T2& B, const T3& C, const T4& D, const std::string& func_name)
  {
  coot_extra_debug_sigprint();

  const no_conv_unwrap<T1> U1(A);
  const no_conv_unwrap<T2> U2(B);
  const no_conv_unwrap<T3> U3(C);
  const no_conv_unwrap<T4> U4(D);

  // check for same number of columns
  const uword A_n_rows = U1.M.n_rows;
  const uword A_n_cols = U1.M.n_cols;
  const uword B_n_rows = U2.M.n_rows;
  const uword B_n_cols = U2.M.n_cols;
  const uword C_n_rows = U3.M.n_rows;
  const uword C_n_cols = U3.M.n_cols;
  const uword D_n_rows = U4.M.n_rows;
  const uword D_n_cols = U4.M.n_cols;

  const uword out_n_cols = A_n_cols + B_n_cols + C_n_cols + D_n_cols;
  const uword out_n_rows = (std::max)((std::max)((std::max)(A_n_rows, B_n_rows), C_n_rows), D_n_rows);

  coot_debug_check( ((A_n_rows != out_n_rows) && ((A_n_rows > 0) || (A_n_cols > 0))), func_name + ": number of rows must be the same" );
  coot_debug_check( ((B_n_rows != out_n_rows) && ((B_n_rows > 0) || (B_n_cols > 0))), func_name + ": number of rows must be the same" );
  coot_debug_check( ((C_n_rows != out_n_rows) && ((C_n_rows > 0) || (C_n_cols > 0))), func_name + ": number of rows must be the same" );
  coot_debug_check( ((D_n_rows != out_n_rows) && ((D_n_rows > 0) || (D_n_cols > 0))), func_name + ": number of rows must be the same" );

  // Shortcut: if there is nothing to do, leave early.
  if (out_n_rows == 0 || out_n_cols == 0)
    {
    out.set_size(out_n_rows, out_n_cols);
    return;
    }

  // We can't have the output be an alias of an input.
  typedef typename no_conv_unwrap<T1>::stored_type UT1;
  typedef typename no_conv_unwrap<T2>::stored_type UT2;
  typedef typename no_conv_unwrap<T3>::stored_type UT3;
  typedef typename no_conv_unwrap<T4>::stored_type UT4;
  alias_wrapper<Mat<eT>, UT1, UT2, UT3, UT4> W(out, U1.M, U2.M, U3.M, U4.M);
  W.use.set_size(out_n_rows, out_n_cols);
  coot_rt_t::join_rows(W.get_dev_mem(false),
                       U1.get_dev_mem(false), A_n_rows, A_n_cols,
                       U2.get_dev_mem(false), B_n_rows, B_n_cols,
                       U3.get_dev_mem(false), C_n_rows, C_n_cols,
                       U4.get_dev_mem(false), D_n_rows, D_n_cols,
                       // subview arguments
                       W.get_row_offset(), W.get_col_offset(), W.get_M_n_rows(),
                       U1.get_row_offset(), U1.get_col_offset(), U1.get_M_n_rows(),
                       U2.get_row_offset(), U2.get_col_offset(), U2.get_M_n_rows(),
                       U3.get_row_offset(), U3.get_col_offset(), U3.get_M_n_rows(),
                       U4.get_row_offset(), U4.get_col_offset(), U4.get_M_n_rows());
  }



template<typename T1, typename T2>
inline
uword
glue_join_rows::compute_n_rows(const Glue<T1, T2, glue_join_rows>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_cols);

  return (std::max)(A_n_rows, B_n_rows);
  }



template<typename T1, typename T2>
inline
uword
glue_join_rows::compute_n_cols(const Glue<T1, T2, glue_join_rows>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(B_n_rows);

  return A_n_cols + B_n_cols;
  }
