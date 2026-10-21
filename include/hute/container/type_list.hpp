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

#include <hute/numeric/integral_constant.hpp>

namespace hute
{

template <class T>
struct type_c {
  using type = T;
};

template <class... T>
struct type_list {};

} // end namespace hute

//
// Specialize tuple-related functionality for hute::type_list
//

#include <tuple>

#include <hute/container/tuple.hpp>

namespace hute
{

template <int I, class... T>
HUTE_HOST_DEVICE constexpr
HUTE_STL_NAMESPACE::tuple_element_t<I, type_list<T...>>
get(type_list<T...>&) noexcept {
  return {};
}
template <int I, class... T>
HUTE_HOST_DEVICE constexpr
HUTE_STL_NAMESPACE::tuple_element_t<I, type_list<T...>>
get(type_list<T...> const& t) noexcept {
  return {};
}

} // end namespace hute

namespace HUTE_STL_NAMESPACE
{

template <class... T>
struct tuple_size<hute::type_list<T...>>
    : HUTE_STL_NAMESPACE::integral_constant<size_t, sizeof...(T)>
{};

template <size_t I, class... T>
struct tuple_element<I, hute::type_list<T...>>
    : hute::type_c<typename HUTE_STL_NAMESPACE::tuple_element<I, HUTE_STL_NAMESPACE::tuple<T...>>::type>
{};

template <class... T>
struct tuple_size<const hute::type_list<T...>>
    : HUTE_STL_NAMESPACE::integral_constant<size_t, sizeof...(T)>
{};

template <size_t I, class... T>
struct tuple_element<I, const hute::type_list<T...>>
    : hute::type_c<typename HUTE_STL_NAMESPACE::tuple_element<I, HUTE_STL_NAMESPACE::tuple<T...>>::type>
{};

} // end namespace std

#ifdef HUTE_STL_NAMESPACE_IS_HIP_STD
namespace std
{

template <class... T>
struct tuple_size<hute::type_list<T...>>
    : HUTE_STL_NAMESPACE::integral_constant<size_t, sizeof...(T)>
{};

template <size_t I, class... T>
struct tuple_element<I, hute::type_list<T...>>
    : hute::type_c<typename HUTE_STL_NAMESPACE::tuple_element<I, HUTE_STL_NAMESPACE::tuple<T...>>::type>
{};

template <class... T>
struct tuple_size<const hute::type_list<T...>>
    : HUTE_STL_NAMESPACE::integral_constant<size_t, sizeof...(T)>
{};

template <size_t I, class... T>
struct tuple_element<I, const hute::type_list<T...>>
    : hute::type_c<typename HUTE_STL_NAMESPACE::tuple_element<I, HUTE_STL_NAMESPACE::tuple<T...>>::type>
{};

} // end namespace std
#endif // HUTE_STL_NAMESPACE_IS_HIP_STD
