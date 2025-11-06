# a = 99  # c
# b = "b"  # 98
# print(chr(a), ord(b))

# c = 0
# d = chr(c)
# print(
#     d, repr(d)
# )  # the first is null, nothing gets printed, the second shows what is really is:  '\x00'

# ## printing many registers
# for i in range(35):
#     print(i, repr(chr(i)))
# print(127, repr(chr(127)))


## encode function in python UTF-8 in unicode standars

def ut8 (a:str ):
    b =a.encode("utf-8") 
    print(f"b ={b}, type of b = {type(b)}, if i apply list={list(b)}  ")

    c = b.decode("utf-8")
    print(f"if i  {c}")
    
if __name__ == '__main__':
    ut8('test')
    