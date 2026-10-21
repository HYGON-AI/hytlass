/***************************************************************************************************
 * Copyright (c) 2023 - 2025 Hygon Information Technology Co., Ltd. All rights reserved.
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
#pragma once

#include <hute/config.hpp>

#include <hute/numeric/integral_constant.hpp>
#include <hute/util/type_traits.hpp>

namespace hute
{

template <class T, size_t N>
struct array
{
  using element_type = T;
  using value_type = remove_cv_t<T>;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using reference = element_type&;
  using const_reference = const element_type&;
  using pointer = element_type*;
  using const_pointer = const element_type*;
  using iterator = pointer;
  using const_iterator = const_pointer;

  HUTE_HOST_DEVICE constexpr
  reference operator[](size_type pos)
  {
    return begin()[pos];
  }

  HUTE_HOST_DEVICE constexpr
  const_reference operator[](size_type pos) const
  {
    return begin()[pos];
  }

  HUTE_HOST_DEVICE constexpr
  reference front()
  {
    return *begin();
  }

  HUTE_HOST_DEVICE constexpr
  const_reference front() const
  {
    return *begin();
  }

  HUTE_HOST_DEVICE constexpr
  reference back()
  {
    // return *rbegin();
    return operator[](N-1);
  }

  HUTE_HOST_DEVICE constexpr
  const_reference back() const
  {
    // return *rbegin();
    return operator[](N-1);
  }

  HUTE_HOST_DEVICE constexpr
  T* data()
  {
    return __elems_;
  }

  HUTE_HOST_DEVICE constexpr
  T const* data() const
  {
    return __elems_;
  }

  HUTE_HOST_DEVICE constexpr
  iterator begin()
  {
    return data();
  }

  HUTE_HOST_DEVICE constexpr
  const_iterator begin() const
  {
    return data();
  }

  HUTE_HOST_DEVICE constexpr
  const_iterator cbegin()
  {
    return begin();
  }

  HUTE_HOST_DEVICE constexpr
  const_iterator cbegin() const
  {
    return begin();
  }

  HUTE_HOST_DEVICE constexpr
  iterator end()
  {
    return data() + size();
  }

  HUTE_HOST_DEVICE constexpr
  const_iterator end() const
  {
    return data() + size();
  }

  HUTE_HOST_DEVICE constexpr
  const_iterator cend()
  {
    return end();
  }

  HUTE_HOST_DEVICE constexpr
  const_iterator cend() const
  {
    return end();
  }

  HUTE_HOST_DEVICE constexpr
  bool empty() const
  {
    return size() == 0;
  }

  HUTE_HOST_DEVICE constexpr
  size_type size() const
  {
    return N;
  }

  HUTE_HOST_DEVICE constexpr
  size_type max_size() const
  {
    return size();
  }

  HUTE_HOST_DEVICE constexpr
  void fill(const T& value)
  {
    for (auto& e : *this) {
      e = value;
    }
  }

  HUTE_HOST_DEVICE constexpr
  void clear()
  {
    fill(T(0));
  }

  HUTE_HOST_DEVICE constexpr
  void swap(array& other)
  {
    using HUTE_STL_NAMESPACE::swap;
    for (size_type i = 0; i < size(); ++i) {
      swap((*this)[i], other[i]);
    }
  }

  element_type __elems_[N];
};


template <class T>
struct array<T, 0>
{
  using element_type = T;
  using value_type = remove_cv_t<T>;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using reference = element_type&;
  using const_reference = const element_type&;
  using pointer = element_type*;
  using const_pointer = const element_type*;
  using const_iterator = const_pointer;
  using iterator = pointer;

  HUTE_HOST_DEVICE constexpr
  reference operator[](size_type pos)
  {
    return begin()[pos];
  }

  HUTE_HOST_DEVICE constexpr
  const_reference operator[](size_type pos) const
  {
    return begin()[pos];
  }

  HUTE_HOST_DEVICE constexpr
  reference front()
  {
    return *begin();
  }

  HUTE_HOST_DEVICE constexpr
  const_reference front() const
  {
    return *begin();
  }

  HUTE_HOST_DEVICE constexpr
  reference back()
  {
    return *begin();
  }

  HUTE_HOST_DEVICE constexpr
  const_reference back() const
  {
    return *begin();
  }

  HUTE_HOST_DEVICE constexpr
  T* data()
  {
    return nullptr;
  }

  HUTE_HOST_DEVICE constexpr
  T const* data() const
  {
    return nullptr;
  }

  HUTE_HOST_DEVICE constexpr
  iterator begin()
  {
    return nullptr;
  }

  HUTE_HOST_DEVICE constexpr
  const_iterator begin() const
  {
    return nullptr;
  }

  HUTE_HOST_DEVICE constexpr
  const_iterator cbegin()
  {
    return nullptr;
  }

  HUTE_HOST_DEVICE constexpr
  const_iterator cbegin() const
  {
    return nullptr;
  }

  HUTE_HOST_DEVICE constexpr
  iterator end()
  {
    return nullptr;
  }

  HUTE_HOST_DEVICE constexpr
  const_iterator end() const
  {
    return nullptr;
  }

  HUTE_HOST_DEVICE constexpr
  const_iterator cend()
  {
    return nullptr;
  }

  HUTE_HOST_DEVICE constexpr
  const_iterator cend() const
  {
    return nullptr;
  }

  HUTE_HOST_DEVICE constexpr
  bool empty() const
  {
    return true;
  }

  HUTE_HOST_DEVICE constexpr
  size_type size() const
  {
    return 0;
  }

  HUTE_HOST_DEVICE constexpr
  size_type max_size() const
  {
    return 0;
  }

  HUTE_HOST_DEVICE constexpr
  void fill(const T& value)
  {}

  HUTE_HOST_DEVICE constexpr
  void clear()
  {}

  HUTE_HOST_DEVICE constexpr
  void swap(array& other)
  {}
};

template <class T, size_t N>
HUTE_HOST_DEVICE constexpr
bool operator==(array<T,N> const& lhs, array<T,N> const& rhs)
{
  for (size_t i = 0; i < N; ++i) {
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }
  return true;
}

template <class T, size_t N>
HUTE_HOST_DEVICE constexpr
void clear(array<T,N>& a)
{
  a.fill(T(0));
}

template <class T, size_t N>
HUTE_HOST_DEVICE constexpr
void fill(array<T,N>& a, T const& value)
{
  a.fill(value);
}

template <class T, size_t N>
HUTE_HOST_DEVICE constexpr
void swap(array<T,N>& a, array<T,N>& b)
{
  a.swap(b);
}

/// @return A hute::array of the elements of @c t in reverse order.
template <class T, size_t N>
HUTE_HOST_DEVICE constexpr
hute::array<T,N> reverse(hute::array<T,N> const& t) 
{
  if constexpr (N == 0u) {
    return t;
  } else {
    hute::array<T,N> t_r{};
    for (size_t k = 0; k < N; ++k) {
      t_r[k] = t[N - k - 1];
    }
    return t_r;
  }
}

} // end hute


//
// Specialize tuple-related functionality for hute::array
//

#include <tuple>

namespace hute
{

template <size_t I, class T, size_t N>
HUTE_HOST_DEVICE constexpr
T& get(array<T,N>& a)
{
  static_assert(I < N, "Index out of range");
  return a[I];
}

template <size_t I, class T, size_t N>
HUTE_HOST_DEVICE constexpr
T const& get(array<T,N> const& a)
{
  static_assert(I < N, "Index out of range");
  return a[I];
}

template <size_t I, class T, size_t N>
HUTE_HOST_DEVICE constexpr
T&& get(array<T,N>&& a)
{
  static_assert(I < N, "Index out of range");
  return hute::move(a[I]);
}

} // end namespace hute

namespace HUTE_STL_NAMESPACE
{

template <class T, size_t N>
struct tuple_size<hute::array<T,N>>
    : HUTE_STL_NAMESPACE::integral_constant<size_t, N>
{};

template <size_t I, class T, size_t N>
struct tuple_element<I, hute::array<T,N>>
{
  using type = T;
};

template <class T, size_t N>
struct tuple_size<hute::array<T,N> const>
    : HUTE_STL_NAMESPACE::integral_constant<size_t, N>
{};

template <size_t I, class T, size_t N>
struct tuple_element<I, hute::array<T,N> const>
{
  using type = T;
};

} // end namespace HUTE_STL_NAMESPACE

#ifdef HUTE_STL_NAMESPACE_IS_HIP_STD
namespace std
{

template <class T, size_t N>
struct tuple_size<hute::array<T,N>>
    : HUTE_STL_NAMESPACE::integral_constant<size_t, N>
{};

template <size_t I, class T, size_t N>
struct tuple_element<I, hute::array<T,N>>
{
  using type = T;
};

template <class T, size_t N>
struct tuple_size<hute::array<T,N> const>
    : HUTE_STL_NAMESPACE::integral_constant<size_t, N>
{};

template <size_t I, class T, size_t N>
struct tuple_element<I, hute::array<T,N> const>
{
  using type = T;
};

} // end namespace std
#endif // HUTE_STL_NAMESPACE_IS_HIP_STD
