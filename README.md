# Homework 4

## Task 1 - Low-/Highpass filter (15 points)
It is your task to implement the `passfilter` function which takes an input signal (`arr`), a cutoff frequency (`cutoff`), the kind of filter (`kind`, default=`lowpass`) and the sampling rate of the signal (`srate`, default=`256`). The options for the `kind` parameter should be `lowpass` (remove frequencies > `cutoff`, let frequencies <= `cutoff` pass) and `highpass` (remove frequencies < `cutoff`, let frequencies >= `cutoff` pass). You can get an array containing the frequency for each element of the array using `fft.fftfreq` (make sure to include the sampling rate here). To remove certain frequencies from the signal (`arr`), transform the signal into the frequency domain (`fft.fft`), set the array elements of the transformed signal less than or greater than (depending on `kind`) the cutoff to 0 and transform back to the time domain (`fft.ifft`).

Hint: You can use NumPy masking with the result of `fft.fftfreq` to determine which frequencies should be set to 0.

## Task 2 - Statistical tests (10 points)
You are supposed to check if two samples from some population have equal mean (`equal_mean`) and equal variance (`equal_var`). To check for equal variance, use a Levene test and return a boolean indicating if it is likely that the two samples (`sample1` and `sample2`) have equal variance. For that, use the p-value from the Levene test and compare it to the `p_threshold`.\
In order to check for equal mean, use T-test for independant samples (`stats.ttest_ind`) and use your `equal_var` function to set the `equal_var` parameter of the T-test function. Again, return a boolean indicating if it is likely that the two samples have equal mean using the `p_threshold`.

Hint: Use the documentation to find out what the tests' null-hypotheses are, with a p-value < p-threshold you can reject the null hypothesis and assume the alternative hypothesis.

## Bonus task - Bloom effect (10 points)
Implement the function `bloom` which should create a kind of [bloom effect](https://en.wikipedia.org/wiki/Bloom_(shader_effect)) for an image that is passed to the function (`img`). In order to achieve this effect first blur the image with a gaussian filter with a sigma of `blur` (`10` by default). Then use a sobel filter on a grayscale version of the blurred image (calculate grayscale with mean over the color channels) to extract the edges. Multiply the image containing the edges with `strength` and use its absolute values to highlight the edges in the original image by simply adding it to each of the color channels. Finally, clip the result to the range [0.0, 1.0] and return it.

> Good luck!
