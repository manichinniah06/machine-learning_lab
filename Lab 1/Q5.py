import random

nums = []

for i in range(25):
    nums.append(random.randint(1,10))

totalsum = sum(nums)

print("Mean : ",totalsum/len(nums)) #Mean

## Median

nums.sort()

n = len(nums)

if n%2 == 1:
    median = nums[n//2]
else:
    median = (nums[n//2 - 1]+nums[n//2])/2

print("Median : ",median)

## Mode

freq = {}

for i in nums:
    if i in freq:
        freq[i] += 1
    else:
        freq[i] = 1

max_freq = 0
mode = 0

for i in freq:
    if freq[i]>max_freq:
        max_freq = freq[i]
        mode = i

print("Mode : ",mode)