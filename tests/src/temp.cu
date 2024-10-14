#include "gtest/gtest.h"

// Temporary test case which does not actually test anything, just to make sure CMake is set up correctly. 

TEST(TestTemp, boolIsTrue) {
    bool b = 1;
    EXPECT_EQ(b, true);
}