# PanelLocalProjections
This repository contains files to implement panel local projections estimation and inference.

Reference: [Almuzara Martin]([url](https://martinalmuzara.com/research.html)) and [Victor Sancibrian]([url](https://sancibrian-v.github.io)), 2024: “[Micro Responses to Macro Shocks.]([url](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr1090.pdf))” Federal Reserve Bank of New York Staff Report, no. 1090.

The **matlab_package** directory contains Matlab code:
  - The main function is **LP_panel.m**
  - The function produces panel local projection estimates, standard errors, confidence intervals and p-values for zero response tests.
  - It can accommodate controls, different types of fixed effects, cumulative impules responses, etc. It also implements the small-sample refinements suggested in the paper.
  - The file **usage_example.m** illustrates how to use it.

The **replication** directory contains the files to replicate the simulation study of Section 4 and the empirical analysis of Section 5.


