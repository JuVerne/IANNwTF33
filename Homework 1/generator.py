def generator():
    output = "meow "
    while True:
        yield output
        output += output


cat = generator()

print(next(cat))
print(next(cat))
print(next(cat))
print(next(cat))
