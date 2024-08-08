import copy

class A():
    def __init__(self, a) -> None:
        self.a = a

class B(A):
    def __init__(self, b) -> None:
        super().__init__(copy.deepcopy(b))
        self.b = b

class C():
    def __init__(self, c) -> None:
        self.c = c
        self.a = A(copy.deepcopy(c))
    
    def deepcopy(self, memo=None):
        if memo is None:
            memo = {}
        memo[id(self.a.a)] = self.a.a
        return copy.deepcopy(self, memo)
    
    # def __deepcopy__(self, memo=None):
    #     if memo is None:    # first time !!!
    #         memo = {}
    #         memo[id(self.a.a)] = self.a.a
    #     else:   # second time !!!
    #         return copy.deepcopy(self, memo)
    #     new_self = copy.copy(self)
    #     memo[id(new_self.a.a)] = new_self.a.a
    #     new_self.a = copy.deepcopy
    #     return C(copy.deepcopy(self.c))

def test_deepcopy():
    x = C({'h': 1, 'i': 2})
    memo = {}
    # memo[id(x.a.a)] = x.a.a
    y = copy.deepcopy(x, memo)
    z = x.deepcopy()
    x.a.a['h'] = 3
    print(f'x.a.a: {x.a.a}')
    print(f'y.a.a: {y.a.a}')
    print(f'z.a.a: {z.a.a}')
    
def test_list():
    x = B([1, 2, 3])
    print(f'x.a: {x.a}, x.b: {x.b}')
    y = copy.copy(x)
    print(f'y.a: {y.a}, y.b: {y.b}')
    y.a[0] *= 2
    print(f'x.a: {x.a}, x.b: {x.b}')
    print(f'y.a: {y.a}, y.b: {y.b}')
    y.a = [4, 5, 6]
    print(f'x.a: {x.a}, x.b: {x.b}')
    print(f'y.a: {y.a}, y.b: {y.b}')
    
    z = copy.deepcopy(x)
    print(f'z.a: {z.a}, z.b: {z.b}')
    z.a[0] *= 2
    print(f'x.a: {x.a}, x.b: {x.b}')
    print(f'z.a: {z.a}, z.b: {z.b}')

def test_dict():
    x = B({'h': 1, 'i': 2})
    y = copy.copy(x)
    y.a['h'] = 3
    print(f'x.a: {x.a}')
    print(f'y.a: {y.a}')
    y.a = copy.copy(x.a)
    y.a['h'] = 4
    print(f'x.a: {x.a}')
    print(f'y.a: {y.a}')
    z = copy.deepcopy(x)
    z.a['h'] = 5
    print(f'x.a: {x.a}')
    print(f'z.a: {z.a}')
    
def main():
    # test_list()
    # test_dict()
    test_deepcopy()

if __name__ == '__main__':
    main()