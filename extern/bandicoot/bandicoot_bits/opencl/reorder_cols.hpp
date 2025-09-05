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


template<typename eT>
inline
void
reorder_cols(dev_mem_t<eT> out, const dev_mem_t<eT> in, const uword n_rows, const dev_mem_t<uword> ordering, const uword out_n_cols)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::reorder_cols(): OpenCL runtime not valid" );

  runtime_t::cq_guard guard;

  runtime_t::adapt_uword local_n_rows(n_rows);
  runtime_t::adapt_uword local_out_n_cols(out_n_cols);
  runtime_t::adapt_uword local_out_mem_offset(out.cl_mem_ptr.offset);
  runtime_t::adapt_uword local_in_mem_offset(in.cl_mem_ptr.offset);

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::reorder_cols);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),            &(out.cl_mem_ptr.ptr)     );
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, local_out_mem_offset.size, local_out_mem_offset.addr );
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(cl_mem),            &(in.cl_mem_ptr.ptr)      );
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, local_in_mem_offset.size,  local_in_mem_offset.addr  );
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, local_n_rows.size,         local_n_rows.addr         );
  status |= coot_wrapper(clSetKernelArg)(kernel, 5, sizeof(cl_mem),            &(ordering.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 6, local_out_n_cols.size,     local_out_n_cols.addr     );

  const size_t global_work_size = size_t(out_n_cols);

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != 0), "coot::opencl::reorder_cols(): couldn't execute kernel" );
  }
