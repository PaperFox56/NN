from math import sin, cos, pi
from random import randint
from arrayViewer import view
from numpy import zeros, array

def blur(a, s, c):
    filter = array([[0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0]])
    out = a.copy()
    for x in range(s):
        for y in range(s):
            x0, y0 = -1, -1
            if x == 0: x0 = 0
            if y == 0: y0 = 0
            x1, y1 = 2, 2
            if x == s-1: x1 = 1
            if y == s-1: y1 = 1
            temp = a[x+x0:x+x1, y+y0:y+y1] * filter[x0+1:x1+1, y0+1:y1+1]
            out[x, y] += c * temp.sum()
            if out[x, y] > 1: out[x, y] = 1
    #print(out)
    return out

def generateCircles(n, s):
    """ Generate n random circles as an array of size s """
    circles = []
    for i in range(n):
        r = randint(2, int(s/2))
        x0, y0 = randint(r, s-r), randint(r, s-r)
        circle = zeros((s, s))
        #print(r, x0, y0)
        for i in range(r*10):
            a = i * pi / (r * 5)
            x, y = int(r*cos(a)), int(r*sin(a))
            try:
                circle[x0+x, y0+y] = 1
            except IndexError: continue
        #circle = blur(circle, s, .15)
        #circle = blur(circle, s, .15)
        circle.resize(s**2)
        circles.append(circle)
    return circles

g = generateCircles(100, 10)
with open("images/circles.py", 'w') as file:
    data = """
from numpy import array
circles = ["""
    for i in range(len(g)):
        data += "\narray(["
        for n in g[i]:
            data += f"{n}, "
        data += "]),\n"
    data += "]"
    file.write(data)
    file.close()
c = g[0].copy()
c.resize(10, 10)
view(c, 10)
