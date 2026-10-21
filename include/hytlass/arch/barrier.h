/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Barrier Operations on SM90+
*/

#pragma once

// for support dtk23.04 and early version
#if __has_include(<hip/hcc_detail/cooperative_groups/details/info.h>)
  #include<hip/hcc_detail/cooperative_groups/details/info.h>
#else
__device__
inline
__attribute__((convergent))
void __syncwarp(unsigned mask=0xffffffff){
    return __syncthreads();
}
#endif

#include <hute/arch/cluster.hpp>
#if defined(__HIP_DEVICE_COMPILE__)
#define HIP_BARRIER_ENABLED 0
#endif

namespace hytlass {
/// @brief
namespace arch {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Enumerates the reserved named barriers to avoid potential conflicts
// This enum class specifies the NamedBarriers reserved by HYTLASS.
enum class ReservedNamedBarriers { 
  EpilogueBarrier = 0,
  TransposeBarrier = 1,
  TransformBarrier = 2,
  StreamkBarrier0 = 3,
  StreamkBarrier1 = 4, 
  FirstUserBarrier = StreamkBarrier1 + 1
};


class NamedBarrier {

  // Data Members:

  // Range = [1 , NUM_THREADS_PER_CTA]
  // Range % warp-size (i.e 32) == 0
  uint32_t const num_threads_;

  // Range : [0, 15]
  // Note that should be set to the final barrier ID, including ReserveNamedBarrierCount should be considered
  uint32_t const id_;

 public:

  // Constructor for HYTLASS developers:
  // effective barrier ID starts from 0
  HYTLASS_DEVICE
  NamedBarrier(uint32_t num_threads, ReservedNamedBarriers reserved_named_barriers)
      : num_threads_(num_threads), id_(static_cast<uint32_t>(reserved_named_barriers)) {}

  // Constructor for HYTLASS users:
  // effective barrier ID starts from ReservedNamedBarrierCount
  HYTLASS_DEVICE
  NamedBarrier(uint32_t num_threads, uint32_t id = 0)
      : num_threads_(num_threads), id_(id + ReservedNamedBarrierCount) {
    HYTLASS_ASSERT(id + ReservedNamedBarrierCount <= HardwareMaxNumNamedBarriers && "Effective barrier_id should not exceed 16.");
  }

  HYTLASS_DEVICE
  void arrive_and_wait() const {
    // Note: The value of id_ is already the final barrier id (set correctly in the constructor).
    NamedBarrier::arrive_and_wait_internal(num_threads_, id_);
  }

  HYTLASS_DEVICE
  void arrive() const {
    // Note: The value of id_ is already the final barrier id (set correctly in the constructor).
    NamedBarrier::arrive_internal(num_threads_, id_);
  }

  HYTLASS_DEVICE
  void sync() const {
    NamedBarrier::arrive_and_wait();
  }

  //  Static variants

  // Calling interface for HYTLASS users: 
  // effective barrier ID starts from ReservedNamedBarrierCount
  HYTLASS_DEVICE
  static void arrive_and_wait(uint32_t num_threads, uint32_t barrier_id) {
    arrive_and_wait_internal(num_threads, barrier_id + ReservedNamedBarrierCount);
  }

  // Calling interface for HYTLASS developers: 
  // effective barrier ID starts from 0
  HYTLASS_DEVICE
  static void arrive_and_wait(uint32_t num_threads, ReservedNamedBarriers reserved_named_barriers) {
    arrive_and_wait_internal(num_threads, static_cast<int>(reserved_named_barriers));
  }

  // Calling interface for HYTLASS users: 
  // effective barrier ID starts from ReservedNamedBarrierCount
  HYTLASS_DEVICE
  static void arrive(uint32_t num_threads, uint32_t barrier_id) {
    arrive_internal(num_threads, barrier_id + ReservedNamedBarrierCount);
  }

  // Calling interface for HYTLASS developers: 
  // effective barrier ID starts from 0
  HYTLASS_DEVICE
  static void arrive(uint32_t num_threads, ReservedNamedBarriers reserved_named_barriers) {
    arrive_internal(num_threads, static_cast<int>(reserved_named_barriers));
  }

  // Calling interface for HYTLASS users: 
  // effective barrier ID starts from ReservedNamedBarrierCount
  HYTLASS_DEVICE
  static void sync(uint32_t num_threads, uint32_t barrier_id) {
    sync_internal(num_threads, barrier_id + ReservedNamedBarrierCount);
  }

  // Calling interface for HYTLASS developers: 
  // effective barrier ID starts from 0
  HYTLASS_DEVICE
  static void sync(uint32_t num_threads, ReservedNamedBarriers reserved_named_barriers) {
    sync_internal(num_threads, static_cast<int>(reserved_named_barriers));
  }

 private:
  HYTLASS_DEVICE
  static void arrive_and_wait_internal(uint32_t num_threads, uint32_t barrier_id) {
#if HIP_BARRIER_ENABLED
    // TODO: enable high-level sync primitives
#elif defined(__HIP_DEVICE_COMPILE__)
    asm volatile ("brkpt;\n" ::);
#endif
  }

  HYTLASS_DEVICE
  static void arrive_internal(uint32_t num_threads, uint32_t barrier_id) {
#if HIP_BARRIER_ENABLED
    // TODO: enable high-level sync primitives
#elif defined(__HIP_DEVICE_COMPILE__)
    asm volatile ("brkpt;\n" ::);
#endif
  }

  HYTLASS_DEVICE
  static void sync_internal(uint32_t num_threads, uint32_t barrier_id) {
    NamedBarrier::arrive_and_wait_internal(num_threads, barrier_id);
  }

 public:
  // Currently we reserve 8 NamedBarriers for HYTLASS' own use cases, 
  // while leaving the renaming for general users.
  static const uint32_t ReservedNamedBarrierCount = static_cast<uint32_t>(ReservedNamedBarriers::FirstUserBarrier);
  static const uint32_t HardwareMaxNumNamedBarriers = 16;

};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Hopper introduces a new cluster-wide barrier which handle with Cluster-wide arrive-wait behaviour.
// This is an extension to the Ampere arrive-wait barriers
// Note : Ampere arrive-wait Barriers have a larger max-arrive count (2^30) than Hopper arrive-wait Barriers (2^20).
struct ClusterBarrier {

  using ValueType = uint64_t;

protected:
  // Can never be initialized - can only be aliased to smem
  ValueType barrier_;

public:

  HYTLASS_DEVICE
  ClusterBarrier() = delete;

  HYTLASS_DEVICE
  void init(uint32_t arrive_count) const {
    ClusterBarrier::init(&this->barrier_, arrive_count);
  }

  HYTLASS_DEVICE
  uint32_t test_wait(uint32_t phase, uint32_t pred=true) const {
    return ClusterBarrier::test_wait(&this->barrier_, phase, pred);
  }

  HYTLASS_DEVICE
  uint32_t try_wait(uint32_t phase) const {
    return ClusterBarrier::try_wait(&this->barrier_, phase);
  }

  HYTLASS_DEVICE
  void wait(uint32_t phase) const {
    ClusterBarrier::wait(&this->barrier_, phase);
  }

  // Barrier arrive on local smem
  HYTLASS_DEVICE
  void arrive() const {
    ClusterBarrier::arrive(&this->barrier_);
  }

  // Remote SMEM arrive with a perdicate (usually done to pick the thread doing the arrive)
  HYTLASS_DEVICE
  void arrive(uint32_t cta_id, uint32_t pred = true ) const {
    ClusterBarrier::arrive(&this->barrier_, cta_id, pred);
  }

  //
  //  Static Versions
  //
  HYTLASS_DEVICE
  static void init(ValueType const* smem_ptr, uint32_t arrive_count) {
#if HIP_BARRIER_ENABLED
    // TODO: enable high-level sync primitives
#elif defined(__HIP_DEVICE_COMPILE__)
    asm volatile ("brkpt;\n" ::);
#endif
  }

  // Static version of wait - in case we don't want to burn a register
  HYTLASS_DEVICE
  static void wait(ValueType const* smem_ptr, uint32_t phase) {
#if HIP_BARRIER_ENABLED
    // TODO: enable high-level sync primitives
#elif defined(__HIP_DEVICE_COMPILE__)
    asm volatile ("brkpt;\n" ::);
#endif
  }

  HYTLASS_DEVICE
  static uint32_t test_wait(ValueType const* smem_ptr, uint32_t phase, uint32_t pred) {
#if HIP_BARRIER_ENABLED
    // TODO: enable high-level sync primitives
#elif defined(__HIP_DEVICE_COMPILE__)
    asm volatile ("brkpt;\n" ::);
#endif
    return 0;
  }

  HYTLASS_DEVICE
  static uint32_t try_wait(ValueType const* smem_ptr, uint32_t phase) {
#if HIP_BARRIER_ENABLED
    // TODO: enable high-level sync primitives
#elif defined(__HIP_DEVICE_COMPILE__)
    asm volatile ("brkpt;\n" ::);
#endif
    return 0;
  }

  // Static Predicated version of the above - in case we know the address.
  HYTLASS_DEVICE
  static void arrive(ValueType const* smem_ptr, uint32_t cta_id, uint32_t pred) {
#if HIP_BARRIER_ENABLED
    // TODO: enable high-level sync primitives
#elif defined(__HIP_DEVICE_COMPILE__)
    asm volatile ("brkpt;\n" ::);
#endif
  }

  // Barrier arrive on local smem
  HYTLASS_DEVICE
  static void arrive(ValueType const* smem_ptr) {
#if HIP_BARRIER_ENABLED
    // TODO: enable high-level sync primitives
#elif defined(__HIP_DEVICE_COMPILE__)
    asm volatile ("brkpt;\n" ::);
#endif
  }

  HYTLASS_DEVICE
  static void invalidate(ValueType const* smem_ptr) {
#if HIP_BARRIER_ENABLED
    // TODO: enable high-level sync primitives
#elif defined(__HIP_DEVICE_COMPILE__)
    asm volatile ("brkpt;\n" ::);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SM90 also introduces a new type of cluster-barrier which supports sync.
// not just based on Arrive Count, but also transaction count (in bytes)
struct ClusterTransactionBarrier : public ClusterBarrier {

  HYTLASS_DEVICE
  ClusterTransactionBarrier() = delete;

  // Performs an arrive operation + expected transaction bytes increment
  HYTLASS_DEVICE
  void arrive_and_expect_tx(uint32_t transaction_bytes) const {
    ClusterTransactionBarrier::arrive_and_expect_tx(&this->barrier_, transaction_bytes);
  }

  // Performs an arrive operation + expected transaction bytes increment
  HYTLASS_DEVICE
  void arrive_and_expect_tx(uint32_t transaction_bytes, uint32_t cta_id, uint32_t pred = 1u) const {
    ClusterTransactionBarrier::arrive_and_expect_tx(&this->barrier_, transaction_bytes , cta_id, pred);
  }

  // Performs an expected transaction bytes increment without doing an arrive operation
  HYTLASS_DEVICE
  void expect_transaction(uint32_t transaction_bytes) const {
    ClusterTransactionBarrier::expect_transaction(&this->barrier_, transaction_bytes);
  }

  // Performs an expected transaction bytes decrement without doing an arrive operation
  HYTLASS_DEVICE
  void complete_transaction(uint32_t dst_cta_id, uint32_t transaction_bytes, uint32_t pred) const {
    ClusterTransactionBarrier::complete_transaction(&this->barrier_, dst_cta_id, transaction_bytes, pred);
  }

  //
  //  Static Versions
  //

  // Performs an arrive operation + expected transaction bytes increment
  HYTLASS_DEVICE
  static void arrive_and_expect_tx(ValueType const* smem_ptr, uint32_t transaction_bytes) {
#if HIP_BARRIER_ENABLED
    // TODO: enable high-level sync primitives
#elif defined(__HIP_DEVICE_COMPILE__)
    asm volatile ("brkpt;\n" ::);
#endif
  }

  // Performs an arrive operation + expected transaction bytes increment for a remote cta_id in a Cluster
  HYTLASS_DEVICE
  static void arrive_and_expect_tx(
      ValueType const* smem_ptr, uint32_t transaction_bytes, uint32_t cta_id, uint32_t pred) {
#if HIP_BARRIER_ENABLED
    // TODO: enable high-level sync primitives
#elif defined(__HIP_DEVICE_COMPILE__)
    asm volatile ("brkpt;\n" ::);
#endif
  }

  // Performs an expected transaction bytes increment without doing an arrive operation
  HYTLASS_DEVICE
  static void expect_transaction(ValueType const* smem_ptr, uint32_t transaction_bytes) {
#if HIP_BARRIER_ENABLED
    // TODO: enable high-level sync primitives
#elif defined(__HIP_DEVICE_COMPILE__)
    asm volatile ("brkpt;\n" ::);
#endif
  }

  // Performs an expected transaction bytes decrement without doing an arrive operation
  HYTLASS_DEVICE
  static void complete_transaction(
      ValueType const* smem_ptr, uint32_t dst_cta_id, uint32_t transaction_bytes, uint32_t pred = 1) {
#if HIP_BARRIER_ENABLED
    // TODO: enable high-level sync primitives
#elif defined(__HIP_DEVICE_COMPILE__)
    asm volatile ("brkpt;\n" ::);
#endif
  }

  //
  // DEPRECATED APIs
  //
  [[deprecated("Use arrive_and_expect_tx instead")]] HYTLASS_DEVICE
  void arrive_and_reset_bytes(uint32_t transaction_bytes) const {
    arrive_and_expect_tx(transaction_bytes);
  }
  [[deprecated("Use arrive_and_expect_tx instead")]] HYTLASS_DEVICE
  void arrive_and_reset_bytes(uint32_t transaction_bytes, uint32_t cta_id) const {
    arrive_and_expect_tx(transaction_bytes, cta_id);
  }
  [[deprecated("Use expect_transaction instead")]] HYTLASS_DEVICE
  void reset_bytes(uint32_t transaction_bytes) const {
    expect_transaction(transaction_bytes);
  }
  [[deprecated("Use arrive_and_expect_tx instead")]] HYTLASS_DEVICE
  static void arrive_and_reset_bytes(ValueType const* smem_ptr, uint32_t transaction_bytes) {
    arrive_and_expect_tx(smem_ptr, transaction_bytes);
  }
  [[deprecated("Use arrive_and_expect_tx instead")]] HYTLASS_DEVICE
  static void arrive_and_reset_bytes(ValueType const* smem_ptr, uint32_t transaction_bytes, uint32_t cta_id, uint32_t pred) {
    arrive_and_expect_tx(smem_ptr, transaction_bytes, cta_id, pred);
  }
  [[deprecated("Use expect_transaction instead")]] HYTLASS_DEVICE
  static void reset_bytes(ValueType const* smem_ptr, uint32_t transaction_bytes) {
    expect_transaction(smem_ptr, transaction_bytes);
  }
  [[deprecated("Use complete_transaction instead")]] HYTLASS_DEVICE
  static void commit(ValueType const* smem_ptr, uint32_t dst_cta_id, uint32_t transaction_bytes, uint32_t pred = 1) {
    complete_transaction(smem_ptr, dst_cta_id, transaction_bytes, pred);
  }
};

// Helps with visibility of barrier init operations across warps / cta / cluster
// Available as a separate function so as to batch inits across barriers and fence once
// Note : It must be composed with an appropriate sync instruction with the right scope
// to ensure visibility eg. __syncthreads() or a cluster_arrive() + cluster_wait()
HYTLASS_DEVICE
void fence_barrier_init() {
#if HIP_BARRIER_ENABLED
  // TODO: enable high-level sync primitives
#elif defined(__HIP_DEVICE_COMPILE__)
  asm volatile ("brkpt;\n" ::);
#endif
}

// Issue a shared memory fence for async operations
HYTLASS_DEVICE
void fence_view_async_shared() {
#if HIP_BARRIER_ENABLED
  // TODO: enable high-level sync primitives
#elif defined(__HIP_DEVICE_COMPILE__)
  asm volatile ("brkpt;\n" ::);
#endif
}

// Arrive on completion of in-flight cp.async operations issued by the calling thread 
HYTLASS_DEVICE
void cpasync_barrier_arrive(uint64_t const* smem_ptr) {
#if HIP_BARRIER_ENABLED
  // TODO: enable high-level sync primitives
#elif defined(__HIP_DEVICE_COMPILE__)
  asm volatile ("brkpt;\n" ::);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
}  // end namespace arch
}  // end namespace hytlass
