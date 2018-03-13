def answer(arr):
    count = 0;
    for i in range(0, len(arr)-2):
        for j in range(i+1, len(arr)-1):
            if max(arr[i], arr[j]) % min(arr[i], arr[j]) == 0:
                for k in range(j+1, len(arr)):
                    if max(arr[k], arr[j]) % min(arr[k], arr[j]) == 0:
                        count += 1;
    return count;

arr = [1, 2, 3, 4, 5, 6]
print answer(arr)