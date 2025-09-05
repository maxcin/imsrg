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



// Assumption: the operation is not diagmat(A) * diagmat(B)!
// Assumption: if the non-diagonal matrix is transposed, it is not also the output!
template<typename eT>
inline
void
mul_diag
  (
  dev_mem_t<eT> C_mem,
  const uword C_n_rows,
  const uword C_n_cols,
  const eT alpha,
  const dev_mem_t<eT> A_mem,
  const bool A_is_diag,
  const bool A_trans,
  const uword A_row_offset,
  const uword A_col_offset,
  const uword A_M_n_rows,
  const dev_mem_t<eT> B_mem,
  const bool B_is_diag,
  const bool B_trans,
  const uword B_row_offset,
  const uword B_col_offset,
  const uword B_M_n_rows
  )
  {
  coot_extra_debug_sigprint();

  cl_mem diag_arg_ptr;
  cl_mem mat_arg_ptr;

  size_t diag_arg_offset;
  size_t mat_arg_offset;

  size_t diag_arg_incr;
  size_t mat_arg_M_n_rows;

  cl_kernel kernel;
  size_t global_work_size;

  if (A_is_diag && !B_is_diag && !B_trans)
    {
    // diagmat(A) * B
    kernel = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::mul_rowwise);

    diag_arg_ptr = A_mem.cl_mem_ptr.ptr;
    mat_arg_ptr  = B_mem.cl_mem_ptr.ptr;

    diag_arg_offset = A_mem.cl_mem_ptr.offset + A_row_offset + A_col_offset * A_M_n_rows;
    mat_arg_offset  = B_mem.cl_mem_ptr.offset + B_row_offset + B_col_offset * B_M_n_rows;

    diag_arg_incr    = A_M_n_rows;
    mat_arg_M_n_rows = B_M_n_rows;

    global_work_size = C_n_cols;
    }
  else if (!A_is_diag && !A_trans && B_is_diag)
    {
    // A * diagmat(B)
    kernel = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::mul_colwise);

    diag_arg_ptr = B_mem.cl_mem_ptr.ptr;
    mat_arg_ptr  = A_mem.cl_mem_ptr.ptr;

    diag_arg_offset = B_mem.cl_mem_ptr.offset + B_row_offset + B_col_offset * B_M_n_rows;
    mat_arg_offset  = A_mem.cl_mem_ptr.offset + A_row_offset + A_col_offset * A_M_n_rows;

    diag_arg_incr    = B_M_n_rows;
    mat_arg_M_n_rows = A_M_n_rows;

    global_work_size = C_n_rows;
    }
  else if (A_is_diag && !B_is_diag && B_trans)
    {
    // diagmat(A) * B'
    kernel = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::mul_rowwise_trans);

    coot_debug_check( mem_overlaps(C_mem, 0, C_n_rows * C_n_cols, B_mem, 0, C_n_rows * C_n_cols), "coot::opencl::mul_diag(): incorrect call, alias and transpose not allowed" );

    diag_arg_ptr = A_mem.cl_mem_ptr.ptr;
    mat_arg_ptr  = B_mem.cl_mem_ptr.ptr;

    diag_arg_offset = A_mem.cl_mem_ptr.offset + A_row_offset + A_col_offset * A_M_n_rows;
    mat_arg_offset  = B_mem.cl_mem_ptr.offset + B_row_offset + B_col_offset * B_M_n_rows;

    diag_arg_incr    = A_M_n_rows;
    mat_arg_M_n_rows = B_M_n_rows;

    global_work_size = C_n_rows;
    }
  else if (!A_is_diag && A_trans && B_is_diag)
    {
    // A' * diagmat(B)
    kernel = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::mul_colwise_trans);

    coot_debug_check( mem_overlaps(C_mem, 0, C_n_rows * C_n_cols, A_mem, 0, C_n_rows * C_n_cols), "coot::opencl::mul_diag(): incorrect call, alias and transpose not allowed" );

    diag_arg_ptr = B_mem.cl_mem_ptr.ptr;
    mat_arg_ptr  = A_mem.cl_mem_ptr.ptr;

    diag_arg_offset = B_mem.cl_mem_ptr.offset + B_row_offset + B_col_offset * B_M_n_rows;
    mat_arg_offset  = A_mem.cl_mem_ptr.offset + A_row_offset + A_col_offset * A_M_n_rows;

    diag_arg_incr    = B_M_n_rows;
    mat_arg_M_n_rows = A_M_n_rows;

    global_work_size = C_n_cols;
    }
  else
    {
    coot_stop_runtime_error("coot::opencl::mul_diag(): incorrect call, at least one matrix must be non-diagonal");
    }

  runtime_t::cq_guard guard;

  runtime_t::adapt_uword cl_diag_arg_offset(diag_arg_offset);
  runtime_t::adapt_uword cl_mat_arg_offset(mat_arg_offset);
  runtime_t::adapt_uword cl_C_offset(C_mem.cl_mem_ptr.offset);
  runtime_t::adapt_uword cl_n_rows(C_n_rows);
  runtime_t::adapt_uword cl_n_cols(C_n_cols);
  runtime_t::adapt_uword cl_diag_arg_incr(diag_arg_incr);
  runtime_t::adapt_uword cl_mat_arg_M_n_rows(mat_arg_M_n_rows);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel,  0, sizeof(cl_mem),           &(C_mem.cl_mem_ptr.ptr) );
  status |= coot_wrapper(clSetKernelArg)(kernel,  1, cl_C_offset.size,         cl_C_offset.addr        );
  status |= coot_wrapper(clSetKernelArg)(kernel,  2, sizeof(cl_mem),           &(diag_arg_ptr)         );
  status |= coot_wrapper(clSetKernelArg)(kernel,  3, cl_diag_arg_offset.size,  cl_diag_arg_offset.addr );
  status |= coot_wrapper(clSetKernelArg)(kernel,  4, cl_diag_arg_incr.size,    cl_diag_arg_incr.addr   );
  status |= coot_wrapper(clSetKernelArg)(kernel,  5, sizeof(cl_mem),           &(mat_arg_ptr)          );
  status |= coot_wrapper(clSetKernelArg)(kernel,  6, cl_mat_arg_offset.size,   cl_mat_arg_offset.addr  );
  status |= coot_wrapper(clSetKernelArg)(kernel,  7, sizeof(eT),               &alpha                  );
  status |= coot_wrapper(clSetKernelArg)(kernel,  8, cl_n_rows.size,           cl_n_rows.addr          );
  status |= coot_wrapper(clSetKernelArg)(kernel,  9, cl_n_cols.size,           cl_n_cols.addr          );
  status |= coot_wrapper(clSetKernelArg)(kernel, 10, cl_mat_arg_M_n_rows.size, cl_mat_arg_M_n_rows.addr);
  coot_check_cl_error(status, "coot::opencl::mul_diag(): couldn't set kernel arguments");

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::mul_diag(): couldn't execute kernel");
  }
