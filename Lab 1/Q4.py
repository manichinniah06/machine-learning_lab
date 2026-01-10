mystring = "hippopotamus"

def maxcharacter(string):
    dict = {}
    for i in string:
        if i in dict:
            dict[i] += 1
        else:
            dict[i] = 1
    max = 0
    max_key = None
    for i in dict:
        if dict[i] > max:
            max = dict[i]
            max_key = i
    return max_key

print(maxcharacter(mystring))