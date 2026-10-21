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

#include <hute/util/type_traits.hpp>
#include <hute/numeric/complex.hpp>

/** C++14 <functional> extensions */

namespace hute {

/**************/
/** Identity **/
/**************/

struct identity {
  template <class T>
  HUTE_HOST_DEVICE constexpr
  decltype(auto) operator()(T&& arg) const {
    return static_cast<T&&>(arg);
  }
};

template <class R>
struct constant_fn {
  template <class... T>
  HUTE_HOST_DEVICE constexpr
  decltype(auto) operator()(T&&...) const {
    return r_;
  }
  R r_;
};

/***********/
/** Unary **/
/***********/

#define HUTE_LEFT_UNARY_OP(NAME,OP)                                  \
  struct NAME {                                                      \
    template <class T>                                               \
    HUTE_HOST_DEVICE constexpr                                       \
    decltype(auto) operator()(T&& arg) const {                       \
      return OP static_cast<T&&>(arg);                                \
    }                                                                \
  }
#define HUTE_RIGHT_UNARY_OP(NAME,OP)                                 \
  struct NAME {                                                      \
    template <class T>                                               \
    HUTE_HOST_DEVICE constexpr                                       \
    decltype(auto) operator()(T&& arg) const {                       \
      return static_cast<T&&>(arg) OP ;                               \
    }                                                                \
  }
#define HUTE_NAMED_UNARY_OP(NAME,OP)                                 \
  struct NAME {                                                      \
    template <class T>                                               \
    HUTE_HOST_DEVICE constexpr                                       \
    decltype(auto) operator()(T&& arg) const {                       \
      return OP (static_cast<T&&>(arg));                              \
    }                                                                \
  }

HUTE_LEFT_UNARY_OP(unary_plus,       +);
HUTE_LEFT_UNARY_OP(negate,           -);
HUTE_LEFT_UNARY_OP(bit_not,          ~);
HUTE_LEFT_UNARY_OP(logical_not,      !);
HUTE_LEFT_UNARY_OP(dereference,      *);
HUTE_LEFT_UNARY_OP(address_of,       &);
HUTE_LEFT_UNARY_OP(pre_increment,   ++);
HUTE_LEFT_UNARY_OP(pre_decrement,   --);

HUTE_RIGHT_UNARY_OP(post_increment, ++);
HUTE_RIGHT_UNARY_OP(post_decrement, --);

HUTE_NAMED_UNARY_OP(abs_fn,           abs);
HUTE_NAMED_UNARY_OP(conjugate, hute::conj);

#undef HUTE_LEFT_UNARY_OP
#undef HUTE_RIGHT_UNARY_OP
#undef HUTE_NAMED_UNARY_OP

template <int Shift_>
struct shift_right_const {
  static constexpr int Shift = Shift_;

  template <class T>
  HUTE_HOST_DEVICE constexpr
  decltype(auto) operator()(T&& arg) const {
    return static_cast<T&&>(arg) >> Shift;
  }
};

template <int Shift_>
struct shift_left_const {
  static constexpr int Shift = Shift_;

  template <class T>
  HUTE_HOST_DEVICE constexpr
  decltype(auto) operator()(T&& arg) const {
    return static_cast<T&&>(arg) << Shift;
  }
};

/************/
/** Binary **/
/************/

#define HUTE_BINARY_OP(NAME,OP)                                      \
  struct NAME {                                                      \
    template <class T, class U>                                      \
    HUTE_HOST_DEVICE constexpr                                       \
    decltype(auto) operator()(T&& lhs, U&& rhs) const {              \
      return static_cast<T&&>(lhs) OP static_cast<U&&>(rhs);           \
    }                                                                \
  }
#define HUTE_NAMED_BINARY_OP(NAME,OP)                                \
  struct NAME {                                                      \
    template <class T, class U>                                      \
    HUTE_HOST_DEVICE constexpr                                       \
    decltype(auto) operator()(T&& lhs, U&& rhs) const {              \
      return OP (static_cast<T&&>(lhs), static_cast<U&&>(rhs));        \
    }                                                                \
  }


HUTE_BINARY_OP(plus,                 +);
HUTE_BINARY_OP(minus,                -);
HUTE_BINARY_OP(multiplies,           *);
HUTE_BINARY_OP(divides,              /);
HUTE_BINARY_OP(modulus,              %);

HUTE_BINARY_OP(plus_assign,         +=);
HUTE_BINARY_OP(minus_assign,        -=);
HUTE_BINARY_OP(multiplies_assign,   *=);
HUTE_BINARY_OP(divides_assign,      /=);
HUTE_BINARY_OP(modulus_assign,      %=);

HUTE_BINARY_OP(bit_and,              &);
HUTE_BINARY_OP(bit_or,               |);
HUTE_BINARY_OP(bit_xor,              ^);
HUTE_BINARY_OP(left_shift,          <<);
HUTE_BINARY_OP(right_shift,         >>);

HUTE_BINARY_OP(bit_and_assign,      &=);
HUTE_BINARY_OP(bit_or_assign,       |=);
HUTE_BINARY_OP(bit_xor_assign,      ^=);
HUTE_BINARY_OP(left_shift_assign,  <<=);
HUTE_BINARY_OP(right_shift_assign, >>=);

HUTE_BINARY_OP(logical_and,         &&);
HUTE_BINARY_OP(logical_or,          ||);

HUTE_BINARY_OP(equal_to,            ==);
HUTE_BINARY_OP(not_equal_to,        !=);
HUTE_BINARY_OP(greater,              >);
HUTE_BINARY_OP(less,                 <);
HUTE_BINARY_OP(greater_equal,       >=);
HUTE_BINARY_OP(less_equal,          <=);

HUTE_NAMED_BINARY_OP(max_fn, hute::max);
HUTE_NAMED_BINARY_OP(min_fn, hute::min);

#undef HUTE_BINARY_OP
#undef HUTE_NAMED_BINARY_OP

/**********/
/** Fold **/
/**********/

#define HUTE_FOLD_OP(NAME,OP)                                        \
  struct NAME##_unary_rfold {                                        \
    template <class... T>                                            \
    HUTE_HOST_DEVICE constexpr                                       \
    auto operator()(T&&... t) const {                                \
      return (t OP ...);                                             \
    }                                                                \
  };                                                                 \
  struct NAME##_unary_lfold {                                        \
    template <class... T>                                            \
    HUTE_HOST_DEVICE constexpr                                       \
    auto operator()(T&&... t) const {                                \
      return (... OP t);                                             \
    }                                                                \
  };                                                                 \
  struct NAME##_binary_rfold {                                       \
    template <class U, class... T>                                   \
    HUTE_HOST_DEVICE constexpr                                       \
    auto operator()(U&& u, T&&... t) const {                         \
      return (t OP ... OP u);                                        \
    }                                                                \
  };                                                                 \
  struct NAME##_binary_lfold {                                       \
    template <class U, class... T>                                   \
    HUTE_HOST_DEVICE constexpr                                       \
    auto operator()(U&& u, T&&... t) const {                         \
      return (u OP ... OP t);                                        \
    }                                                                \
  }

HUTE_FOLD_OP(plus,                 +);
HUTE_FOLD_OP(minus,                -);
HUTE_FOLD_OP(multiplies,           *);
HUTE_FOLD_OP(divides,              /);
HUTE_FOLD_OP(modulus,              %);

HUTE_FOLD_OP(plus_assign,         +=);
HUTE_FOLD_OP(minus_assign,        -=);
HUTE_FOLD_OP(multiplies_assign,   *=);
HUTE_FOLD_OP(divides_assign,      /=);
HUTE_FOLD_OP(modulus_assign,      %=);

HUTE_FOLD_OP(bit_and,              &);
HUTE_FOLD_OP(bit_or,               |);
HUTE_FOLD_OP(bit_xor,              ^);
HUTE_FOLD_OP(left_shift,          <<);
HUTE_FOLD_OP(right_shift,         >>);

HUTE_FOLD_OP(bit_and_assign,      &=);
HUTE_FOLD_OP(bit_or_assign,       |=);
HUTE_FOLD_OP(bit_xor_assign,      ^=);
HUTE_FOLD_OP(left_shift_assign,  <<=);
HUTE_FOLD_OP(right_shift_assign, >>=);

HUTE_FOLD_OP(logical_and,         &&);
HUTE_FOLD_OP(logical_or,          ||);

HUTE_FOLD_OP(equal_to,            ==);
HUTE_FOLD_OP(not_equal_to,        !=);
HUTE_FOLD_OP(greater,              >);
HUTE_FOLD_OP(less,                 <);
HUTE_FOLD_OP(greater_equal,       >=);
HUTE_FOLD_OP(less_equal,          <=);

#undef HUTE_FOLD_OP

/**********/
/** Meta **/
/**********/

template <class Fn, class Arg>
struct bound_fn {

  template <class T>
  HUTE_HOST_DEVICE constexpr
  decltype(auto)
  operator()(T&& arg) {
    return fn_(arg_, static_cast<T&&>(arg));
  }

  Fn fn_;
  Arg arg_;
};

template <class Fn, class Arg>
HUTE_HOST_DEVICE constexpr
auto
bind(Fn const& fn, Arg const& arg) {
  return bound_fn<Fn,Arg>{fn, arg};
}

} // end namespace hute
