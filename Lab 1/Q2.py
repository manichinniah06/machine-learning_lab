## Question2

mylist = [5,3,8,1,0,4]

def maxmin(list):
    if len(list) < 3:
        return "Range determination not possible"
    max = list[0]
    min = list[0]
    for i in list:
        if i > max:
            max = i
        if i < min:
            min = i
    return max - min
    
print(maxmin(mylist))