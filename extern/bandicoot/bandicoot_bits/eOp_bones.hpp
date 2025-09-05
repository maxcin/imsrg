// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2008-2023 Conrad Sanderson (https://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
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


// eOps are elementwise operations with up to one scalar argument.

template<typename T1, typename eop_type>
class eOp : public Base< typename T1::elem_type, eOp<T1, eop_type> >
  {
  public:

  typedef typename T1::elem_type                                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;

  static constexpr bool is_row = T1::is_row;
  static constexpr bool is_col = T1::is_col;
  static constexpr bool is_xvec = T1::is_xvec;

  coot_aligned const SizeProxy<T1> m;

  coot_aligned       elem_type  aux;          // storage of auxiliary data, user defined format
  coot_aligned       uword      aux_uword_a;  // storage of auxiliary data, uword format
  coot_aligned       uword      aux_uword_b;  // storage of auxiliary data, uword format

  inline         ~eOp();
  inline explicit eOp(const T1& in_m);
  inline          eOp(const T1& in_m, const elem_type in_aux);
  inline          eOp(const T1& in_m, const uword in_aux_uword_a, const uword in_aux_uword_b);
  inline          eOp(const T1& in_m, const elem_type in_aux, const uword in_aux_uword_a, const uword in_aux_uword_b);

  coot_inline uword get_n_rows() const;
  coot_inline uword get_n_cols() const;
  coot_inline uword get_n_elem() const;
  };
