// Copyright 2021-2025 Ryan Curtin (http://www.ratml.org)
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



template<typename eT1, typename eT2>
inline
void
max(dev_mem_t<eT2> dest,
    const dev_mem_t<eT1> src,
    const uword n_rows,
    const uword n_cols,
    const uword dim,
    const bool post_conv_apply,
    // subview arguments
    const uword dest_offset,
    const uword dest_mem_incr,
    const uword src_row_offset,
    const uword src_col_offset,
    const uword src_M_n_rows)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::max(): cuda runtime not valid" );

  CUfunction kernel;
  if (dim == 0)
    {
    kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(post_conv_apply ? twoway_kernel_id::max_colwise_conv_post : twoway_kernel_id::max_colwise_conv_pre);
    }
  else
    {
    kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(post_conv_apply ? twoway_kernel_id::max_rowwise_conv_post : twoway_kernel_id::max_rowwise_conv_pre);
    }

  const uword src_offset = src_row_offset + src_col_offset * src_M_n_rows;

  const eT2* dest_ptr = dest.cuda_mem_ptr + dest_offset;
  const eT1*  src_ptr =  src.cuda_mem_ptr + src_offset;

  const void* args[] = {
      &dest_ptr,
      &src_ptr,
      (uword*) &n_rows,
      (uword*) &n_cols,
      (uword*) &dest_mem_incr,
      (uword*) &src_M_n_rows };

  const kernel_dims dims = one_dimensional_grid_dims((dim == 0) ? n_cols : n_rows);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::max(): cuLaunchKernel() failed");
  }



/**
 * Compute the maximum of all elements in `mem`.
 * This is basically identical to `accu()`.
 */
template<typename eT>
inline
eT
max_vec(dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::max_vec(): cuda runtime not valid" );

  CUfunction k = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::max);
  CUfunction k_small = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::max_small);

  return generic_reduce<eT, eT>(mem, n_elem, "max_vec", k, k_small, std::make_tuple(/* no extra args */));
  }



/**
 * Compute the maximum of elements in a Cube in each column.
 * This particular operation cannot be done with any of the matrix min kernels.
 */
template<typename eT1, typename eT2>
inline
void
max_cube_col(dev_mem_t<eT2> dest,
             const dev_mem_t<eT1> src,
             const uword n_rows,
             const uword n_cols,
             const uword n_slices,
             const bool post_conv_apply)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::max_cube_col(): CUDA runtime not valid" );

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(post_conv_apply ? twoway_kernel_id::max_cube_col_conv_post : twoway_kernel_id::max_cube_col_conv_pre);

  const void* args[] = {
      &(dest.cuda_mem_ptr),
      &(src.cuda_mem_ptr),
      (uword*) &n_rows,
      (uword*) &n_cols,
      (uword*) &n_slices };

  const kernel_dims dims = two_dimensional_grid_dims(n_rows, n_slices);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::max_cube_col(): cuLaunchKernel() failed");

  }
