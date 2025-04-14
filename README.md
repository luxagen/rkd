# RKD

A fast and efficient file tree comparison tool written in Rust.

RKD is short for "RotKraken Diff".

<https://github.com/luxagen/rkd>

## Overview

RKD is a utility for comparing two file trees or log files and identifying differences between them. It's designed to work with the [RotKraken](https://github.com/luxagen/rotkraken) (rk) tool for generating file hashes and can efficiently detect file creations, deletions, modifications, and moves/copies between directory trees.

## Features

- Compare two directory trees or log files
- Detect file creations, deletions, modifications, and moves/copies
- Efficiently handle large directory structures
- Filter out paths containing specific substrings
- Display common path prefixes or full paths

## Installation

### Prerequisites

- Rust and Cargo
- The [RotKraken](https://github.com/luxagen/rotkraken) (rk) tool for generating file hashes when comparing directory trees

### Building from Source

```bash
cargo build --release
```

The compiled binary will be available at `target/release/rkd`.

## Usage

```
rkd [OPTIONS] <left> <right>
```

### Arguments

- `<left>`: Left-hand (before) tree or log file
- `<right>`: Right-hand (after) tree or log file

### Options

- `-x, --exclude <EXCLUDE>`: Ignore paths containing this substring
- `-t, --time`: Print phase timings to stderr
- `-P, --no-prefix`: Omit common path prefix and show full paths
- `-h, --help`: Print help information
- `-V, --version`: Print version information

### Examples

Compare two directory trees:
```bash
rkd /path/to/directory1 /path/to/directory2
```

Compare two log files:
```bash
rkd log1.txt log2.txt
```

Compare using stdin for one side:
```bash
some_command | rkd - /path/to/directory
```

Exclude certain paths:
```bash
rkd --exclude=temp --exclude=cache /path/to/directory1 /path/to/directory2
```

## Links

- RotKraken:
    - GitHub: [rotkraken](https://github.com/luxagen/rotkraken)
    - Homepage: [http://www.luxagen.com/product/rotkraken](http://www.luxagen.com/product/rotkraken)
