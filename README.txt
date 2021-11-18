# -*- coding: utf-8 -*-
This is the code for my LIN5111 language identification project. My method consists of a hierachical classifier
which successively applies sorting by script, multinomial naive bayes classification, and feedforward neural networks.

The script_id.py mainly exists to declutter the main tedreader.py file and
is not intended to be run by itself. The main has the loading the corpus step
outside of the main function so that main() can be called on different 
subsets of the corpus without reloading the whole thing. 
Testing is by far the most 
time consuming part of running the program, lasting several hours when using half the corpus. 
I've set the code to run on about 5%, but the corpus size kwarg can control the fraction used.

As the dataset is over 1 GB,
(and per the instructions), I have not included it in this zip folder, but it can be 
downloaded at https://github.com/UKPLab/sentence-transformers/blob/master/docs/datasets/TED2020.md
Here is the paper describing the corpus: https://arxiv.org/pdf/2004.09813.pdf
