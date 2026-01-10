## Question1
mylist = [2,7,4,1,3,6]

def checksum(list):
    result = []
    for i in range(len(list)):
        for j in range(i+1,len(list)):
            if list[i] + list[j] == 10:
                    result.append((list[i],list[j]))
    
    return result

print(checksum(mylist))