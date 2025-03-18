# seq-img
A script created for sequential generation of images

Simply run train.py, and follow instructions

Experiments show that columnwise learning is the fastest/best, tied with rowwise. Classic Patch mode learns global features faster, but inner content takes far longer to converge (slightly faster if LSTM is used, but those have their own issues)

Will add more details soon
