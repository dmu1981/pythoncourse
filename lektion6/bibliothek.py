def fakultaet(n):
    if n <= 1:
        return 1
    
    return n * fakultaet(n-1)