namespace utilities
{
    template <typename Functor>
    void timeit(const Functor& f)
    {
        const auto begin = std::chrono::steady_clock::now();
        f();
        const auto end = std::chrono::steady_clock::now();
        std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << std::endl;
    }


    template <typename T, size_t N>
    size_t countof(const T(&v)[N])
    {
        //static_assert(std::is_same_v<T, int>, "Only sizeof int vector");
        return N;
    }

}