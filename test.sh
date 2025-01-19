#!/bin/bash
DIR=/btr/a/hash-logs
cargo build -r
echo GOOD
target/release/rkd <(cd "$DIR" && sudo git show f9778302c05b85f1b92f77c04c0deb5d09af7f9b~1:md5.txt) <(cd "$DIR" && sudo git show f9778302c05b85f1b92f77c04c0deb5d09af7f9b:md5.txt) 2>&1 | sort | tee test-good.txt
echo BAD
sudo target/release/rkd / / 2>&1 | sort | tee test-bad.txt
