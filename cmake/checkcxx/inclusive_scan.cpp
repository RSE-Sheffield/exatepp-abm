// Test if std::inclusive_scan is supported or not by the current compiler c++ stdlib.
#include <numeric>
#include <vector>

int main(int argc, char * argv[]) {
    std::vector<float> vec = {{0.f, 1.f, 2.f, 3.f}};
    std::inclusive_scan(vec.begin(), vec.end(), vec.begin());
    return 0;
}