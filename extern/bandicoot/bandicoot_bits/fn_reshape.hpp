// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2022      Ryan Curtin (http://www.ratml.org/)
// Copyright 2008-2016 Conrad Sanderson (https://conradsanderson.id.au)
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



template<typename T1>
coot_warn_unused
inline
typename enable_if2<
  is_coot_type<T1>::value,
  const Op<T1, op_reshape>
>::result
reshape(const T1& x, const uword in_n_rows, const uword in_n_cols)
  {
  coot_extra_debug_sigprint();

  return Op<T1, op_reshape>(x, in_n_rows, in_n_cols);
  }



template<typename T1>
coot_warn_unused
inline
typename enable_if2<
  is_coot_type<T1>::value,
  const Op<T1, op_reshape>
>::result
reshape(const T1& x, const SizeMat& s)
  {
  coot_extra_debug_sigprint();

  return Op<T1, op_reshape>(x, s.n_rows, s.n_cols);
  }
