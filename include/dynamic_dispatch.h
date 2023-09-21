#include <type_traits>

template <class... Variants>
struct dynamic_dispatch : Variants... {
    using Variants::operator()...;
};
template <class... Variants>
dynamic_dispatch(Variants...) -> dynamic_dispatch<Variants...>;
