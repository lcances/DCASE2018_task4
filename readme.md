# Stack

### Different convolution scheme
Each audio last for 10 seconds and localisation must be done with an accuracy of 200ms

###### Stack
- s1 | 9x500 ---(1,2)---> 9x250 ---(1,5)---> 9x50
- s2 | 9x500 ---(3,2)---> 3x250 ---(3x5)---> 1x50

##### Mel
- m1 | 64x500 ---(4,1)---> 16x500 ---(4,1)---> 4x500 ---(4,1)---> 1x500
- m2 | 64x500 ---(4,1)---> 16x500 ---(4,5)---> 4x100 ---(4,2)---> 1x50
 