#include <iostream>

template <std::size_t ...>
struct gSequence
 { };

template <std::size_t N, std::size_t ... Next>
struct gSequenceHelper : public gSequenceHelper<N-1U, N-1U, Next...>
 { };

template <std::size_t ... Next>
struct gSequenceHelper<0U, Next ... >
 { using type = gSequence<Next ... >; };

template <std::size_t N>
using makeGSequence = typename gSequenceHelper<N>::type;
