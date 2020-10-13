#@HEADER
# ******************************************************************************
#
#  CP-CALS: Software for computing the Canonical Polyadic Decomposition using
#  the Concurrent Alternating Least Squares Algorithm.
#
#  Copyright (c) 2020, Christos Psarras
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its
#     contributors may be used to endorse or promote products derived from
#     this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ******************************************************************************
#@HEADER

SET(tests als cals)

function(create_test test source)

    SET(BLA_VEN MKL)
    if (${WITH_BLIS})
        SET(BLA_VEN BLIS)
    endif ()
    if (${WITH_OPENBLAS})
        SET(BLA_VEN OPENBLAS)
    endif ()

    SET(name ${test}_${BLA_VEN})
    ADD_EXECUTABLE(${name} ${source}.cpp)
    if (GTEST_DEPENDENCY)
        ADD_DEPENDENCIES(${name} ${GTEST_DEPENDENCY})
    endif ()

    TARGET_LINK_LIBRARIES(${name} PRIVATE CALS_${BLA_VEN})
    TARGET_LINK_LIBRARIES(${name} PRIVATE ${GTEST_LINK_FLAGS})

    ADD_TEST(NAME ${name} COMMAND ${name})

endfunction(create_test)

foreach (test_dir ${tests})
    ADD_SUBDIRECTORY(${test_dir})
endforeach ()
