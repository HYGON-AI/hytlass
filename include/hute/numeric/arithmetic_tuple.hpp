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

#include <hute/container/tuple.hpp>
#include <hute/numeric/integral_constant.hpp>
#include <hute/algorithm/functional.hpp>
#include <hute/algorithm/tuple_algorithms.hpp>
#include <hute/util/type_traits.hpp>

namespace hute
{

template <class... T>
struct ArithmeticTuple : tuple<T...>
{
  template <class... U>
  HUTE_HOST_DEVICE constexpr
  ArithmeticTuple(ArithmeticTuple<U...> const& u)
    : tuple<T...>(static_cast<tuple<U...> const&>(u)) {}

  template <class... U>
  HUTE_HOST_DEVICE constexpr
  ArithmeticTuple(tuple<U...> const& u)
    : tuple<T...>(u) {}

  template <class... U>
  HUTE_HOST_DEVICE constexpr
  ArithmeticTuple(U const&... u)
    : tuple<T...>(u...) {}
};

template <class... T>
struct is_tuple<ArithmeticTuple<T...>> : true_type {};

template <class... Ts>
struct is_flat<ArithmeticTuple<Ts...>> : is_flat<tuple<Ts...>> {};

template <class... T>
HUTE_HOST_DEVICE constexpr
auto
make_arithmetic_tuple(T const&... t) {
  return ArithmeticTuple<T...>(t...);
}

template <class T>
HUTE_HOST_DEVICE constexpr
auto
as_arithmetic_tuple(T const& t) {
  if constexpr (is_tuple<T>::value) {
    return detail::tapply(t, [](auto const& x){ return as_arithmetic_tuple(x); },
                          [](auto const&... a){ return make_arithmetic_tuple(a...); },
                          tuple_seq<T>{});
  } else {
    return t;
  }
}

//
// Numeric operators
//

// Addition
template <class... T, class... U>
HUTE_HOST_DEVICE constexpr
auto
operator+(ArithmeticTuple<T...> const& t, ArithmeticTuple<U...> const& u) {
  constexpr int R = hute::max(int(sizeof...(T)), int(sizeof...(U)));
  return transform_apply(append<R>(t,Int<0>{}), append<R>(u,Int<0>{}), plus{}, [](auto const&... a){ return make_arithmetic_tuple(a...); });
}

template <class... T, class... U>
HUTE_HOST_DEVICE constexpr
auto
operator+(ArithmeticTuple<T...> const& t, tuple<U...> const& u) {
  return t + ArithmeticTuple<U...>(u);
}

template <class... T, class... U>
HUTE_HOST_DEVICE constexpr
auto
operator+(tuple<T...> const& t, ArithmeticTuple<U...> const& u) {
  return ArithmeticTuple<T...>(t) + u;
}

// Subtraction
template <class... T, class... U>
HUTE_HOST_DEVICE constexpr
auto
operator-(ArithmeticTuple<T...> const& t, ArithmeticTuple<U...> const& u) {
  constexpr int R = hute::max(int(sizeof...(T)), int(sizeof...(U)));
  return transform_apply(append<R>(t,Int<0>{}), append<R>(u,Int<0>{}), minus{}, [](auto const&... a){ return make_arithmetic_tuple(a...); });
}

template <class... T, class... U>
HUTE_HOST_DEVICE constexpr
auto
operator-(ArithmeticTuple<T...> const& t, tuple<U...> const& u) {
  return t - ArithmeticTuple<U...>(u);
}

template <class... T, class... U>
HUTE_HOST_DEVICE constexpr
auto
operator-(tuple<T...> const& t, ArithmeticTuple<U...> const& u) {
  return ArithmeticTuple<T...>(t) - u;
}

// Negation
template <class... T>
HUTE_HOST_DEVICE constexpr
auto
operator-(ArithmeticTuple<T...> const& t) {
  return transform_apply(t, negate{}, [](auto const&... a){ return make_arithmetic_tuple(a...); });
}

//
// Special cases
//

template <auto t, class... U>
HUTE_HOST_DEVICE constexpr
ArithmeticTuple<U...> const&
operator+(C<t>, ArithmeticTuple<U...> const& u) {
  static_assert(t == 0, "Arithmetic tuple op+ error!");
  return u;
}

template <class... T, auto u>
HUTE_HOST_DEVICE constexpr
ArithmeticTuple<T...> const&
operator+(ArithmeticTuple<T...> const& t, C<u>) {
  static_assert(u == 0, "Arithmetic tuple op+ error!");
  return t;
}

template <auto t, class... U>
HUTE_HOST_DEVICE constexpr
ArithmeticTuple<U...> const&
operator-(C<t>, ArithmeticTuple<U...> const& u) {
  static_assert(t == 0, "Arithmetic tuple op- error!");
  return -u;
}

template <class... T, auto u>
HUTE_HOST_DEVICE constexpr
ArithmeticTuple<T...> const&
operator-(ArithmeticTuple<T...> const& t, C<u>) {
  static_assert(u == 0, "Arithmetic tuple op- error!");
  return t;
}

//
// ArithmeticTupleIterator
//

template <class ArithTuple>
struct ArithmeticTupleIterator
{
  using value_type   = ArithTuple;
  using element_type = ArithTuple;
  using reference    = ArithTuple;

  ArithTuple coord_;

  HUTE_HOST_DEVICE constexpr
  ArithmeticTupleIterator(ArithTuple const& coord = {}) : coord_(coord) {}

  HUTE_HOST_DEVICE constexpr
  ArithTuple const& operator*() const { return coord_; }

  template <class Coord>
  HUTE_HOST_DEVICE constexpr
  auto operator[](Coord const& c) const { return *(*this + c); }

  template <class Coord>
  HUTE_HOST_DEVICE constexpr
  auto operator+(Coord const& c) const {
    return ArithmeticTupleIterator<decltype(coord_ + c)>(coord_ + c);
  }
};

template <class Tuple>
HUTE_HOST_DEVICE constexpr
auto
make_inttuple_iter(Tuple const& t) {
  return ArithmeticTupleIterator(as_arithmetic_tuple(t));
}

template <class T0, class T1, class... Ts>
HUTE_HOST_DEVICE constexpr
auto
make_inttuple_iter(T0 const& t0, T1 const& t1, Ts const&... ts) {
  return make_inttuple_iter(hute::make_tuple(t0, t1, ts...));
}

//
// ArithmeticTuple "basis" elements
//   A ScaledBasis<T,N> is a (at least) rank-N+1 ArithmeticTuple:
//      (_0,_0,...,T,_0,...)
//   with value T in the Nth mode

template <class T, int N>
struct ScaledBasis : private tuple<T>
{
  HUTE_HOST_DEVICE constexpr
  ScaledBasis(T const& t = {}) : tuple<T>(t) {}

  HUTE_HOST_DEVICE constexpr
  decltype(auto) value()       { return get<0>(static_cast<tuple<T>      &>(*this)); }
  HUTE_HOST_DEVICE constexpr
  decltype(auto) value() const { return get<0>(static_cast<tuple<T> const&>(*this)); }

  HUTE_HOST_DEVICE static constexpr
  auto mode() { return Int<N>{}; }
};

template <class T>
struct is_scaled_basis : false_type {};
template <class T, int N>
struct is_scaled_basis<ScaledBasis<T,N>> : true_type {};

template <class T, int N>
struct is_integral<ScaledBasis<T,N>> : true_type {};

// Get the scalar T out of a ScaledBasis
template <class SB>
HUTE_HOST_DEVICE constexpr auto
basis_value(SB const& e)
{
  if constexpr (is_scaled_basis<SB>::value) {
    return basis_value(e.value());
  } else {
    return e;
  }
  HUTE_GCC_UNREACHABLE;
}

// Apply the N... pack to another Tuple
template <class SB, class Tuple>
HUTE_HOST_DEVICE constexpr auto
basis_get(SB const& e, Tuple const& t)
{
  if constexpr (is_scaled_basis<SB>::value) {
    return basis_get(e.value(), get<SB::mode()>(t));
  } else {
    return t;
  }
  HUTE_GCC_UNREACHABLE;
}

namespace detail {

template <class T, int... I>
HUTE_HOST_DEVICE constexpr
auto
to_atuple_i(T const& t, seq<I...>) {
  return make_arithmetic_tuple((void(I),Int<0>{})..., t);
}

} // end namespace detail

// Turn a ScaledBases<T,N> into a rank-N+1 ArithmeticTuple
//    with N prefix 0s:  (_0,_0,...N...,_0,T)
template <class T, int N>
HUTE_HOST_DEVICE constexpr
auto
as_arithmetic_tuple(ScaledBasis<T,N> const& t) {
  return detail::to_atuple_i(as_arithmetic_tuple(t.value()), make_seq<N>{});
}

namespace detail {

template <int... Ns>
struct Basis;

template <>
struct Basis<> {
  using type = Int<1>;
};

template <int N, int... Ns>
struct Basis<N,Ns...> {
  using type = ScaledBasis<typename Basis<Ns...>::type, N>;
};

} // end namespace detail

// Shortcut for writing ScaledBasis<ScaledBasis<ScaledBasis<Int<1>, N0>, N1>, ...>
// E<>    := _1
// E<0>   := (_1,_0,_0,...)
// E<1>   := (_0,_1,_0,...)
// E<0,0> := ((_1,_0,_0,...),_0,_0,...)
// E<0,1> := ((_0,_1,_0,...),_0,_0,...)
// E<1,0> := (_0,(_1,_0,_0,...),_0,...)
// E<1,1> := (_0,(_0,_1,_0,...),_0,...)
template <int... N>
using E = typename detail::Basis<N...>::type;

template <class Shape>
HUTE_HOST_DEVICE constexpr
auto
make_basis_like(Shape const& shape)
{
  if constexpr (is_integral<Shape>::value) {
    return Int<1>{};
  } else {
    // Generate bases for each rank of shape
    return transform(tuple_seq<Shape>{}, shape, [](auto I, auto si) {
      // Generate bases for each rank of si and add an i on front
      using I_type = decltype(I);
      return transform_leaf(make_basis_like(si), [](auto e) {
        // MSVC has trouble capturing variables as constexpr,
        // so that they can be used as template arguments.
        // This is exactly what the code needs to do with i, unfortunately.
        // The work-around is to define i inside the inner lambda,
        // by using just the type from the enclosing scope.
        constexpr int i = I_type::value;
        return ScaledBasis<decltype(e), i>{};
      });
    });
  }

  HUTE_GCC_UNREACHABLE;
}

//
// Arithmetic
//

template <class T, int M, class U>
HUTE_HOST_DEVICE constexpr
auto
safe_div(ScaledBasis<T,M> const& b, U const& u)
{
  auto t = safe_div(b.value(), u);
  return ScaledBasis<decltype(t),M>{t};
}

template <class T, int M, class U>
HUTE_HOST_DEVICE constexpr
auto
shape_div(ScaledBasis<T,M> const& b, U const& u)
{
  auto t = shape_div(b.value(), u);
  return ScaledBasis<decltype(t),M>{t};
}

// Equality
template <class T, int N, class U, int M>
HUTE_HOST_DEVICE constexpr
auto
operator==(ScaledBasis<T,N> const& t, ScaledBasis<U,M> const& u) {
  return bool_constant<M == N>{} && t.value() == u.value();
}

// Not equal to anything else
template <class T, int N, class U>
HUTE_HOST_DEVICE constexpr
false_type
operator==(ScaledBasis<T,N> const&, U const&) {
  return {};
}

template <class T, class U, int M>
HUTE_HOST_DEVICE constexpr
false_type
operator==(T const&, ScaledBasis<U,M> const&) {
  return {};
}

// Abs
template <class T, int N>
HUTE_HOST_DEVICE constexpr
auto
abs(ScaledBasis<T,N> const& e) {
  return ScaledBasis<decltype(abs(e.value())),N>{abs(e.value())};
}

// Multiplication
template <class A, class T, int N>
HUTE_HOST_DEVICE constexpr
auto
operator*(A const& a, ScaledBasis<T,N> const& e) {
  auto r = a * e.value();
  return ScaledBasis<decltype(r),N>{r};
}

template <class T, int N, class B>
HUTE_HOST_DEVICE constexpr
auto
operator*(ScaledBasis<T,N> const& e, B const& b) {
  auto r = e.value() * b;
  return ScaledBasis<decltype(r),N>{r};
}

// Addition
template <class T, int N, class U, int M>
HUTE_HOST_DEVICE constexpr
auto
operator+(ScaledBasis<T,N> const& t, ScaledBasis<U,M> const& u) {
  return as_arithmetic_tuple(t) + as_arithmetic_tuple(u);
}

template <class T, int N, class... U>
HUTE_HOST_DEVICE constexpr
auto
operator+(ScaledBasis<T,N> const& t, ArithmeticTuple<U...> const& u) {
  return as_arithmetic_tuple(t) + u;
}

template <class... T, class U, int M>
HUTE_HOST_DEVICE constexpr
auto
operator+(ArithmeticTuple<T...> const& t, ScaledBasis<U,M> const& u) {
  return t + as_arithmetic_tuple(u);
}

template <auto t, class U, int M>
HUTE_HOST_DEVICE constexpr
auto
operator+(C<t>, ScaledBasis<U,M> const& u) {
  static_assert(t == 0, "ScaledBasis op+ error!");
  return u;
}

template <class T, int N, auto u>
HUTE_HOST_DEVICE constexpr
auto
operator+(ScaledBasis<T,N> const& t, C<u>) {
  static_assert(u == 0, "ScaledBasis op+ error!");
  return t;
}

//
// Display utilities
//

template <class ArithTuple>
HUTE_HOST_DEVICE void print(ArithmeticTupleIterator<ArithTuple> const& iter)
{
  printf("ArithTuple"); print(iter.coord_);
}

template <class T, int N>
HUTE_HOST_DEVICE void print(ScaledBasis<T,N> const& e)
{
  print(e.value()); printf("@%d", N);
}

template <class ArithTuple>
HUTE_HOST std::ostream& operator<<(std::ostream& os, ArithmeticTupleIterator<ArithTuple> const& iter)
{
  return os << "ArithTuple" << iter.coord_;
}

template <class T, int N>
HUTE_HOST std::ostream& operator<<(std::ostream& os, ScaledBasis<T,N> const& e)
{
  return os << e.value() << "@" << N;
}

} // end namespace hute


namespace HUTE_STL_NAMESPACE
{

template <class... T>
struct tuple_size<hute::ArithmeticTuple<T...>>
  : HUTE_STL_NAMESPACE::integral_constant<size_t, sizeof...(T)>
{};

template <size_t I, class... T>
struct tuple_element<I, hute::ArithmeticTuple<T...>>
  : HUTE_STL_NAMESPACE::tuple_element<I, HUTE_STL_NAMESPACE::tuple<T...>>
{};

template <class... T>
struct tuple_size<const hute::ArithmeticTuple<T...>>
  : HUTE_STL_NAMESPACE::integral_constant<size_t, sizeof...(T)>
{};

template <size_t I, class... T>
struct tuple_element<I, const hute::ArithmeticTuple<T...>>
  : HUTE_STL_NAMESPACE::tuple_element<I, const HUTE_STL_NAMESPACE::tuple<T...>>
{};

} // end namespace HUTE_STL_NAMESPACE

#ifdef HUTE_STL_NAMESPACE_IS_HIP_STD
namespace std
{

template <class... T>
struct tuple_size<hute::ArithmeticTuple<T...>>
  : HUTE_STL_NAMESPACE::integral_constant<size_t, sizeof...(T)>
{};

template <size_t I, class... T>
struct tuple_element<I, hute::ArithmeticTuple<T...>>
  : HUTE_STL_NAMESPACE::tuple_element<I, HUTE_STL_NAMESPACE::tuple<T...>>
{};

template <class... T>
struct tuple_size<const hute::ArithmeticTuple<T...>>
  : HUTE_STL_NAMESPACE::integral_constant<size_t, sizeof...(T)>
{};

template <size_t I, class... T>
struct tuple_element<I, const hute::ArithmeticTuple<T...>>
  : HUTE_STL_NAMESPACE::tuple_element<I, const HUTE_STL_NAMESPACE::tuple<T...>>
{};

} // end namespace std
#endif // HUTE_STL_NAMESPACE_IS_HIP_STD
