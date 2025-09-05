// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2023 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2023 National ICT Australia (NICTA)
// Copyright 2021-2023 Marcus Edel (http://kurg.org)
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



class glue_min
  : public traits_glue_or
  {
  public:

  //
  // for operations on matrices
  //

  template<typename out_eT, typename T1, typename T2>
  inline static void apply(Mat<out_eT>& out, const Glue<T1, T2, glue_min>& X);

  template<typename T1, typename T2> inline static uword compute_n_rows(const Glue<T1, T2, glue_min>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols);
  template<typename T1, typename T2> inline static uword compute_n_cols(const Glue<T1, T2, glue_min>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols);

  //
  // for operations on cubes
  //

  template<typename out_eT, typename T1, typename T2>
  inline static void apply(Cube<out_eT>& out, const GlueCube<T1, T2, glue_min>& X);

  template<typename T1, typename T2> inline static uword   compute_n_rows(const GlueCube<T1, T2, glue_min>& glue, const uword A_n_rows, const uword A_n_slices, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices);
  template<typename T1, typename T2> inline static uword   compute_n_cols(const GlueCube<T1, T2, glue_min>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices);
  template<typename T1, typename T2> inline static uword compute_n_slices(const GlueCube<T1, T2, glue_min>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices);
  };
