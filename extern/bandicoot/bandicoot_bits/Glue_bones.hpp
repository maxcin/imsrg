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



template<typename T1, typename T2, typename glue_type>
class Glue : public Base<typename T1::elem_type, Glue<T1, T2, glue_type> >
  {
  public:

  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;

  static constexpr bool is_row = (is_same_type<glue_type,glue_times>::value && T1::is_row);
  static constexpr bool is_col = (is_same_type<glue_type,glue_times>::value && T2::is_col);
  static constexpr bool is_xvec = (is_row || is_col);

  coot_inline  Glue(const T1& in_A, const T2& in_B);
  coot_inline  Glue(const T1& in_A, const T2& in_B, const uword in_aux_uword);
  coot_inline ~Glue();

  const T1&   A;
  const T2&   B;
        uword aux_uword;
  };
