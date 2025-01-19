#!/bin/bash
target/release/rkd <(cd /btr/int/hash-logs && sudo git show f9778302c05b85f1b92f77c04c0deb5d09af7f9b~1:md5.txt) <(cd /btr/int/hash-logs && sudo git show f9778302c05b85f1b92f77c04c0deb5d09af7f9b:md5.txt) 2>&1 | sort | tee testcase.txt
