# Channel Coding Project - Incremental Redundancy

**COMM B504 - Spring 2025**  
**Authors:** Mahmoud Elfil, Omar Emad, Youssef Sabry

## Overview

This MATLAB project simulates a full channel coding system using convolutional codes and incremental redundancy (IR). It evaluates transmission performance under a binary symmetric channel (BSC) by processing video data, encoding it with various rates, simulating noisy transmission, and decoding with or without redundancy.

### Key Features

- Convolutional encoding: G = [171, 133], constraint length = 7
- Punctured rates: 8/9 to 1/2
- Incremental redundancy with progressive retransmissions
- Viterbi decoding with traceback
- BER and throughput measurement across BSC error probabilities
- Six decoded video showcases as per project instructions

## Folder Structure

```text
project-root/
│
├── code/                  # MATLAB source code
│   └── main.m             # Main simulation script
│
├── data/
│   └── highway.avi        # Input video
├── docs/
│   ├── Project Description.pdf
│   └── Project Guidelines.pdf
│
├── plots/                 # Output figures (auto-generated)
│   ├── BER_plot.png
│   └── Throughput_plot.png
│
├── videos/                # Output decoded videos (auto-generated)
│   ├── decoded_none_p1e-03.avi
│   ├── decoded_1_2_p1e-03.avi
│   ├── decoded_ir_p1e-03.avi
│   ├── decoded_none_p1e-01.avi
│   ├── decoded_1_2_p1e-01.avi
│   └── decoded_ir_p1e-01.avi
│
├── startup.m              # Initializes path and working directory
└── README.md              # This file
```

## How to Run

1. Open MATLAB
2. Set the current folder to `/code`
3. Run:

```matlab
run(fullfile('..','startup.m'));
main
```

All results (plots and videos) will be generated automatically.

## Outputs

### Plots

- `BER_plot.png`: Bit Error Rate (BER) vs channel error probability
- `Throughput_plot.png`: Throughput vs channel error probability (for IR only)

### Videos

Six `.avi` files showing decoded video quality under different conditions:

| Filename                 | Scheme               | Channel Error |
|--------------------------|----------------------|---------------|
| decoded_none_p1e-03.avi  | Uncoded              | p = 0.001     |
| decoded_1_2_p1e-03.avi   | Rate 1/2             | p = 0.001     |
| decoded_ir_p1e-03.avi    | Incremental Red.     | p = 0.001     |
| decoded_none_p1e-01.avi  | Uncoded              | p = 0.1       |
| decoded_1_2_p1e-01.avi   | Rate 1/2             | p = 0.1       |
| decoded_ir_p1e-01.avi    | Incremental Red.     | p = 0.1       |

## Authors

- Mahmoud Elfil
- Omar Emad
- Youssef Sabry
