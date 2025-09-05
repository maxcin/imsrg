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



template<typename T1, typename op_type, bool condition>
class Op_traits {};




template<typename T1, typename op_type>
class Op_traits<T1, op_type, true>
  {
  public:

  static constexpr bool is_row  = op_type::template traits<T1>::is_row;
  static constexpr bool is_col  = op_type::template traits<T1>::is_col;
  static constexpr bool is_xvec = op_type::template traits<T1>::is_xvec;
  };



template<typename T1, typename op_type>
class Op_traits<T1, op_type, false>
  {
  public:

  static constexpr bool is_row  = false;
  static constexpr bool is_col  = false;
  static constexpr bool is_xvec = false;
  };



template<typename T1, typename op_type>
class Op
  : public Base< typename T1::elem_type, Op<T1, op_type> >
  , public Op_traits<T1, op_type, has_nested_op_traits<op_type>::value>
  {
  public:

  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;

  inline explicit Op(const T1& in_m);
  inline          Op(const T1& in_m, const elem_type in_aux);
  inline          Op(const T1& in_m, const elem_type in_aux,         const uword in_aux_uword_a, const uword in_aux_uword_b);
  inline          Op(const T1& in_m, const uword     in_aux_uword_a, const uword in_aux_uword_b);
  inline          Op(const T1& in_m, const uword     in_aux_uword_a, const uword in_aux_uword_b, const uword in_aux_uword_c, const char junk);
  // `junk` is ignored and exists to disambiguate specializations; pass any constant char
  inline          Op(const T1& in_m, const char      junk,           const elem_type in_aux,     const elem_type in_aux_b);

  inline         ~Op();

  coot_aligned const T1&       m;
  coot_aligned       elem_type aux;
  coot_aligned       elem_type aux_b;
  coot_aligned       uword     aux_uword_a;
  coot_aligned       uword     aux_uword_b;
  coot_aligned       uword     aux_uword_c;
  };
