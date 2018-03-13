l=[4, 3, 1, 7, 8];
def answer(l, t):
    start = 0; end = 0; sum = l[0];
    out = [-1, -1];
    '''
    for i in range(1, len(l)):
        while (sum > t and start < i - 1):
            sum -= l[start];
            start+=1;

        if (sum == t):
            out[0] = start;
            out[1] = i - 1;
            return out;
        if (i < len(l)):
            sum += l[i];
    return out
'''
    while (end + 1 < len(l)):
        while (sum < t and end + 1 < len(l)):
            end+=1;
            sum += l[end];

        while (sum > t and end > start):
            sum -= l[start];
            start+=1;
        if (sum == t):
            out[0] = start;
            out[1] = end;
            return out;
        if (end == start and sum > t):
            sum -= l[start];
            start +=1;
            end +=1;
            sum += l[end];
    return out
output = answer(l, 15)
print output
