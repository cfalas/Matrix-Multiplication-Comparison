# Matrix Multiplication Speed Test

In this test, I will be comparing a custom matrix multiplication function using lists, along with numpy.matmul, and trying to use the numba @jit decorator to speed my function up

## Methodology

The script that I used is the file multiply.py. Each matrix was generated randomly with numbers from 1 to 1000

## Results

In all of the graphs, execution time is plotted against the size of the matrices to be multiplied.
For the first figure, we have a comparison between my custom function and numpy for values of N from 0..500. It is quite clear that numpy completely crushes my function, with numpy looking completely linear (if we zoom in, we can see that this is not the case) in comparison with my function.

In the second figure, I added the @jit decorator. As we can see, for the first few runs of the function, numba was a lot slower, since it had to compile everything. However, after that, since it was running from cache, it was much faster, in most cases slightly outperforming the same function without the jit decorator

The third figure looks identical to the second, but upon closer inspection it looks like the jit decorator made things faster. This is because I specified the data types inside the decorated, which spead it up.

In the fourth figure, we can see the same results on a logarithmic scale, where we can see more clearly the differences between the methods, but with a smaller domain of N, but at smaller intervals

In the fifth figure, we again see a logarithmic graph, but with the same 0..500 domain as the original graphs.

## Possible explanations

I believe that numpy is so much faster because of the fact that even though it is used in python, it is originally written and executed in C, which is a low level, compiled language, which is much faster than Python. In most cases, the difference between Python and C is not so significant, but because of the large number of calculations, the difference adds up.

Something else that may need more time in Python is the variable-sized integers, which is needed to have arbitrarily large integers. 