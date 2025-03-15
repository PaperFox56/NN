from numpy import array, floor
from tkinter import Canvas

def view(a, s):
    size = a.shape
    can = Canvas(width=size[0]*s, height=size[1]*s, bg="#ffffff")
    for j in range(size[0]):
        for i in range(size[1]):
            c = a[j, i]
            c = hex(int((1-c)*255))[2:]
            if len(c) == 1: c = "0"+c
            x, y = i*s, j*s
            can.create_rectangle(x, y, x+s, y+s, width=0, fill=f"#{c}{c}{c}")
            #print(c, end=" ")
        #print()
    print
    can.pack()
    can.mainloop()


if __name__ == "__main__":
    from ones import ones
    a = ones[0].copy()
    a.resize(6, 6)
    view(a, 20)
