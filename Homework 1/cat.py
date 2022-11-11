class Cat:

    def __init__(self, a):
        self.name = a

    def name(self):
        print(self.name)

    def greeting(self, b):
        print("Hello, I am " + self.name)
        print("I see you " + b)


a = Cat("Garfield")
b = Cat("Siam")

# a.greeting("test")
# b.greeting("safa")

# print(a)
