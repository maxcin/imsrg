// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
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



// Base * scalar
template<typename T1>
coot_inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_scalar_times> >::result
operator*
(const T1& X, const typename T1::elem_type k)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_scalar_times>(X, k);
  }



// scalar * Base
template<typename T1>
coot_inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_scalar_times> >::result
operator*
(const typename T1::elem_type k, const T1& X)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_scalar_times>(X, k);  // NOTE: order is swapped
  }



// scalar * trans(T1)
template<typename T1>
coot_inline
const Op<T1, op_htrans2>
operator*
(const typename T1::elem_type k, const Op<T1, op_htrans>& X)
  {
  coot_extra_debug_sigprint();

  return Op<T1, op_htrans2>(X.m, k);
  }



// trans(T1) * scalar
template<typename T1>
coot_inline
const Op<T1, op_htrans2>
operator*
(const Op<T1, op_htrans>& X, const typename T1::elem_type k)
  {
  coot_extra_debug_sigprint();

  return Op<T1, op_htrans2>(X.m, k);
  }



// multiplication of Base objects with same element type
template<typename T1, typename T2>
coot_inline
typename
enable_if2
  <
  is_coot_type<T1>::value && is_coot_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value && !(resolves_to_diagmat<T1>::value || resolves_to_diagmat<T2>::value),
  const Glue<T1, T2, glue_times>
  >::result
operator*
(const T1& X, const T2& Y)
  {
  coot_extra_debug_sigprint();

  return Glue<T1, T2, glue_times>(X, Y);
  }



// multiplication of Base objects with different element types
template<typename T1, typename T2>
inline
typename
enable_if2
  <
  (is_coot_type<T1>::value && is_coot_type<T2>::value && (is_same_type<typename T1::elem_type, typename T2::elem_type>::no)),
  const Glue< T1, T2, glue_mixed_times >
  >::result
operator*
  (
  const T1& X,
  const T2& Y
  )
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT1;
  typedef typename T2::elem_type eT2;

  promote_type<eT1,eT2>::check();

  return Glue<T1, T2, glue_mixed_times>( X, Y );
  }



// Base * diagmat, diagmat * Base, diagmat * diagmat
template<typename T1, typename T2>
coot_inline
typename
enable_if2
  <
  (is_coot_type<T1>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value && (resolves_to_diagmat<T1>::value || resolves_to_diagmat<T2>::value)),
  const Glue<T1, T2, glue_times_diag>
  >::result
operator*
  (
  const T1& X,
  const T2& Y
  )
  {
  coot_extra_debug_sigprint();

  return Glue<T1, T2, glue_times_diag>(X, Y);
  }
