class Fibonnaci:
    def __init__(self, maximum):
        self.a, self.b = 1, 1
        self.maximum = maximum
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.a, self.b = self.a + self.b, self.a
        
        if self.a > self.maximum:
            raise StopIteration()
        
        return self.a
    
for i in Fibonnaci(20):
    print(i)

lst = list(Fibonnaci(1000))    
print(lst)