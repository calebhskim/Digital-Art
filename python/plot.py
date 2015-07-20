from PIL import Image
import numpy as np
import math
import time
import itertools
from graph import graph
from polynomial import polynomial
import random
import images2gif
import os, errno
from win32gui import GetDC, SetPixel
from hilbert3d import hilbert3d2
from math import pi, sin, cos, log

def colors0(n):
    return [rbow0(2*pi*k/n, 1) for k in range(n)]

def colors(n):
    return [rbow(2*pi*k/n, 1) for k in range(n)]

def coloradd(c1, c2):
    return coloravg(c1, c2, (1, 1))

def colorscale(color, mult):
    return coloravg(color, 0, (mult, 0))

def coloravg(c1, c2, (w1, w2)=(0.5, 0.5)):
    return 0x010000*min(int(c1/0x010000*w1 + c2/0x010000*w2), 0x0000FF) \
         + 0x000100*min(int(c1%0x010000/0x000100*w1 + c2%0x010000/0x000100*w2), 0x0000FF) \
         + min(int(c1%0x000100*w1 + c2%0x000100*w2), 0x0000FF)

def rgb(r=0, g=0, b=0, permute=0):
    (r,g,b) = list(itertools.permutations((r,g,b)))[permute]
    return min(255, int(r)) + 0x000100*min(255, int(g)) + 0x010000*min(255, int(b))

def hsl(h, s, l):
    c = (1 - abs(2*l - 1)) * s
    m = l - c/2
    H = (h / (pi / 3)) % 6
    x = c*(1 - abs(H%2 - 1))
    if h is None:
        (r, g, b) = (m, m, m)
    elif 0 <= H < 1:
        (r, g, b) = (c + m, x + m, m)
    elif 1 <= H < 2:
        (r, g, b) = (x + m, c + m, m)
    elif 2 <= H < 3:
        (r, g, b) = (m, c + m, x + m)
    elif 3 <= H < 4:
        (r, g, b) = (m, x + m, c + m)
    elif 4 <= H < 5:
        (r, g, b) = (x + m, m, c + m)
    else:
        (r, g, b) = (c + m, m, x + m)
    return rgb(255*r, 255*g, 255*b)

class plotdata(object):
    def __init__(self, vw, color=0):
        self.vw = vw
        self.color = color
        self.num = [0]*(vw.dimx*vw.dimy)
        if color != None: self.data = [color]*(vw.dimx*vw.dimy)
        else: self.data = [0]*(vw.dimx*vw.dimy)

    def getpixel((i, j)):
        return self.data[j*self.vw.dimx + i]

    def putpixel(self, (x, y), color, add=False, avg=False):
        if 0 <= x < self.vw.dimx and 0 <= y < self.vw.dimy:
            i = y*self.vw.dimx + x
            if not add:
                self.data[i] = color
            elif avg:
                self.data[i] = coloravg(self.data[i], color, (self.num[i]/(self.num[i] + 1.0), 1/(self.num[i] + 1.0)))
                self.num[i] += 1
            else:
                self.data[i] = coloradd(self.data[i], color)
            if SCREEN and x < 1366 and y < 768:
                SetPixel(DC, x, y, self.data[i])

    def putpoint(self, (x, y), color, add=False, avg=False):
        self.putpixel((self.vw.xcartpx(x), self.vw.ycartpx(y)), color, add, avg)

    def save(self, filename, save=True):
        plot = Image.new("RGB",(self.vw.dimx,self.vw.dimy))
        if self.color is None:
            data2 = [0]*(self.vw.dimx*self.vw.dimy)
            for i in range(len(self.data)):
                if self.data[i] != 0:
                    data2[i] = rbow(min(2*pi, self.data[i]), 1)
        else:
            data2 = self.data
        plot.putdata(data2)
        if save:
            plot.save(dir+filename+fileextension)
            print "saved", filename+fileextension, "to", dir, "at", time.asctime()
        return plot

    def line(self, (x1, y1), (x2, y2), color, add=False, avg=False):
        if self.vw.inwindow((x1, y1)) and self.vw.inwindow((x2, y2)):
            x1, x2 = self.vw.xcartpx(x1), self.vw.xcartpx(x2)
            y1, y2 = self.vw.ycartpx(y1), self.vw.ycartpx(y2)
            if abs(x2 - x1) > abs(y2 - y1):
                if x1 > x2: (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
                for i in range(x1, x2 + 1):
                    for c in range(-self.vw.thick, self.vw.thick+1):
                        for d in range(-self.vw.thick, self.vw.thick+1):
                            if c**2 + d**2 <= self.vw.thick**2 and x1 != x2:
                                self.putpixel((i + d, y1 + (i-x1)*(y2-y1)/(x2-x1) + c), color, add, avg)
            else:
                if y1 > y2: (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
                for i in range(y1, y2 + 1):
                    for c in range(-self.vw.thick, self.vw.thick+1):
                        for d in range(-self.vw.thick, self.vw.thick+1):
                             if c**2 + d**2 <= self.vw.thick**2 and y1 != y2:
                                self.putpixel((x1 + (i-y1)*(x2-x1)/(y2-y1) + d, i + c), color, add, avg)

    def circle(self, (x0, y0), r, color, add=False, avg=False, pts=400):
        t = 0
        while t < 2*pi:
            u = t + 2*pi/pts
            self.line((x0 + r*cos(t), y0 + r*sin(t)), (x0 + r*cos(u), y0 + r*sin(u)), color, add, avg)
            t = u

    def circumcircle(self, (ax, ay), (bx, by), (cx, cy), color, add=False, avg=False, pts=400):
        d = 2 * (ax*(by - cy) + bx*(cy - ay) + cx*(ay - by))
        if d == 0:
            self.line((ax,ay),(bx,by), color, add, avg)
            self.line((cx,cy),(bx,by), color, add, avg)
        else:
            x0 = ((ax**2 + ay**2)*(by - cy) + (bx**2 + by**2)*(cy - ay) + (cx**2 + cy**2)*(ay - by))/d
            y0 = ((ax**2 + ay**2)*(cx - bx) + (bx**2 + by**2)*(ax - cx) + (cx**2 + cy**2)*(bx - ax))/d
            r = math.sqrt((ax - x0)**2 + (ay - y0)**2)
            self.circle((x0, y0), r, color, add, avg, pts)

class viewwindow(object):
    def __init__(self, xmin=-10, xmax=10, ymin=10, ymax=10, tmin=0, tmax=2*pi, tstep=pi/2500, dimx=500, dimy=500, thick=1, axisx=False, axisy=False):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.tmin = tmin
        self.tmax = tmax
        self.tstep = tstep
        self.dimx = dimx
        self.dimy = dimy
        self.thick = thick
        self.axisx = axisx
        self.axisy = axisy

    def autoparam(self, xlist, ylist):
        self.xmin = min([min([x(t) for t in [self.tstep*t for t in range(int(self.tmax/self.tstep))]]) for x in xlist])
        self.ymin = min([min([y(t) for t in [self.tstep*t for t in range(int(self.tmax/self.tstep))]]) for y in ylist])
        self.xmax = max([max([x(t) for t in [self.tstep*t for t in range(int(self.tmax/self.tstep))]]) for x in xlist])
        self.ymax = max([max([y(t) for t in [self.tstep*t for t in range(int(self.tmax/self.tstep))]]) for y in ylist])
        self.xmin, self.xmax = self.xmin - .1*(self.xmax-self.xmin), self.xmax + .1*(self.xmax-self.xmin)
        self.ymin, self.ymax = self.ymin - .1*(self.ymax-self.ymin), self.ymax + .1*(self.ymax-self.ymin)
        if self.xmax - self.xmin > self.ymax - self.ymin:
            self.dimy = int(self.dimy*(self.ymax-self.ymin)/(self.xmax-self.xmin))
        else:
            self.dimx = int(self.dimx*(self.xmax-self.xmin)/(self.ymax-self.ymin))

    def autograph(self, funclist):
        factor=self.dimy/(self.ymax-self.ymin)
        self.ymin = min([min(map(f,[t*(xmin+(self.xmax-self.xmin)/self.dimx) for t in range(dimx)])) for f in funclist])
        self.ymax = max([max(map(f,[t*(xmin+(self.xmax-self.xmin)/self.dimx) for t in range(dimx)])) for f in funclist])
        self.ymin, self.ymax = self.ymin - .1*(self.ymax-self.ymin), self.ymax + .1*(self.ymax-self.ymin)
        dimy = int((self.ymax-self.ymin)*factor)

    def autocomplex(self, *pts):
        self.xmin = min([z.real for z in pts])
        self.ymin = min([z.imag for z in pts])
        self.xmax = max([z.real for z in pts])
        self.ymax = max([z.imag for z in pts])
        self.xmin, self.xmax = self.xmin - .5*(self.xmax-self.xmin), self.xmax + .5*(self.xmax-self.xmin)
        self.ymin, self.ymax = self.ymin - .5*(self.ymax-self.ymin), self.ymax + .5*(self.ymax-self.ymin)
        if self.xmax - self.xmin > self.ymax - self.ymin:
            self.dimy = int(self.dimy*(self.ymax-self.ymin)/(self.xmax-self.xmin))
        else:
            self.dimx = int(self.dimx*(self.xmax-self.xmin)/(self.ymax-self.ymin))

    def inwindow(self, (x,y)):
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def ypxcart(self, y):
        return self.ymax - y*(self.ymax-self.ymin+0.0)/self.dimy

    def ycartpx(self, y):
        if self.inwindow((self.xmin,y)):
            return round(self.dimy*(self.ymax-y)/(self.ymax-self.ymin))

    def xpxcart(self, x):
        return x*(self.xmax-self.xmin+0.0)/self.dimx+self.xmin

    def xcartpx(self, x):
        if self.inwindow((x,self.ymin)):
            return round(self.dimx*(x-self.xmin)/(self.xmax-self.xmin))

    def pxiterator(self):
        for j in xrange(self.dimy):
            for i in xrange(self.dimx):
                yield (i, j)

    def cartiterator(self):
        for j in xrange(self.dimy):
            for i in xrange(self.dimx):
                yield (self.xpxcart(i), self.ypxcart(j))

    def complexiterator(self):
        for j in xrange(self.dimy):
            for i in xrange(self.dimx):
                yield self.xpxcart(i) + self.ypxcart(j)*1j

def round(x):
    if abs(x - int(x)) < .5:
        return int(x)
    else:
        return int(x + math.copysign(1, x))

def yaxis(vw, data, color=0x555555):
    if vw.inwindow((0,vw.ymax)) and vw.xmax != 0 and vw.xmin != 0:
        for i in range(vw.dimy):
            data.putpixel((vw.xcartpx(0), i), color)
    return data

def xaxis(vw, data, color=0x555555):
    if vw.inwindow((vw.xmax, 0)) and vw.ymin != 0 and vw.xmin != 0:
        for i in range(vw.dimx):
            data.putpixel((i, vw.ycartpx(0)), color)
    return data

def init(vw, color=0x000000):
    data = plotdata(vw, color)
    if vw.axisx:
        data = xaxis(vw, data)
    if vw.axisy:
        data = yaxis(vw, data)
    return data

def suffix(vw, long=False):
    if long:
        return "-["+str(vw.xmin)+","+str(vw.xmax)+"]x["+str(vw.ymin)+","+str(vw.ymax)+"]"
    return ""

def rbow(t, freq=1.5):
    return rgb(127*math.sin(freq*t)+128,
               127*math.sin(freq*t-2*pi/3)+128,
               127*math.sin(freq*t+2*pi/3)+128)

def rbow0(t, freq=1.5):
    return hsl(t*freq, 1, .5)

def diffquo(f, h):
    return lambda x: (f(x+h)-f(x-h))/(2*h)

def newton(f, z0, df=None, n=10, epsilon=1e-10, h=.0001):
    if df is None:
        df = diffquo(f, h)
    if n == 0 or abs(df(z0)) < epsilon:
        return z0
    else:
        return newton(f, z0 - f(z0)/df(z0), df, n-1, epsilon, h)

def gradient(f, (x0,y0), h=.0001):
    return (diffquo(lambda x: f(x,y0), h)(x0), diffquo(lambda y: f(x0,y),h)(y0))

def closest(rootlist, approx, d=lambda z1,z2: abs(z1-z2)):
    min = d(rootlist[0], approx)
    mindex = 0;
    for i in range(1, len(rootlist)):
        if d(rootlist[i], approx) < min:
            min = d(rootlist[i], approx)
            mindex = i
    return mindex

def rootsofunity(n, phi=0):
    return [cos(2*k*pi/n + phi) + 1j*sin(2*k*pi/n + phi) for k in range(n)]

def rootsquadratic(a,b,c):
    return [(-b + math.sqrt(b**2 - 4*a*c))/2/a, (-b - math.sqrt(b**2 - 4*a*c))/2/a]

def colorcube(func="dfs"):
    if not func.startswith("hilbert"):
        return list(eval("graph.grid3d(64, 64, 64)."+func+"((0,0,0))"))
    else:
        return list(hilbert3d(6))

def hilbert(n):
    s = "A"
    for i in range(n):
        s2 = ""
        for c in s:
            if c == 'A':
                s2 += "-BF+AFA+FB-"
            elif c == 'B':
                s2 += "+AF-BFB-FA+"
            else:
                s2 += c
        s = s2
    (x, y) = (0, 0)
    (dx, dy) = (1, 0)
    yield (x, y)
    for c in s:
        if c == '+':
            (dx, dy) = (dy, -dx)
        elif c == '-':
            (dx, dy) = (-dy, dx)
        elif c == 'F':
            (x, y) = (x + dx, y + dy)
            yield (x, y)

def hilbert3d(n):
    if n == 6 and hilbert3d2 is not None:
        for i in hilbert3d2():
            yield i
        return
    s = 'X'
    for i in range(n):
        s2 = ""
        for c in s:
            if c == 'X':
                s2 += "^<XF^<XFX-F^>>XFX&F+>>XFX-F>X->"
            else:
                s2 += c
        s = s2
    (x, y, z) = (0, 0, 0)
    hlu = [[1, 0, 0],[0, 1, 0],[0, 0, 1]]
    yield (x, y, z)
    for c in s:
        if c == 'F':
            (x, y, z) = (x + hlu[0][0], y + hlu[1][0], z + hlu[2][0])
            yield (x, y, z)
        elif c == '+': hlu = multmat(hlu, [[0,-1,0],[1,0,0],[0,0,1]]) #yaw(90)
        elif c == '-': hlu = multmat(hlu, [[0,1,0],[-1,0,0],[0,0,1]]) #yaw(-90)
        elif c == '>': hlu = multmat(hlu, [[1,0,0],[0,0,-1],[0,1,0]]) #roll(90)
        elif c == '<': hlu = multmat(hlu, [[1,0,0],[0,0,1],[0,-1,0]]) #roll(-90)
        elif c == '&': hlu = multmat(hlu, [[0,0,1],[0,1,0],[-1,0,0]]) #pitch(-90)
        elif c == '^': hlu = multmat(hlu, [[0,0,-1],[0,1,0],[1,0,0]]) #pitch(90)

def multmat(m1, m2):
    prod = []
    for i in range(len(m1)):
        prod.append([0]*len(m2[0]))
    for i, row in enumerate(m1):
        for j, col in enumerate(transpose(m2)):
            prod[i][j] = sum([a*b for a,b in zip(row, col)])
    return prod

def transpose(m):
    m2 = []
    for i in range(len(m[0])):
        m2.append([0]*len(m))
    for i in range(len(m)):
        for j in range(len(m[0])):
            m2[j][i] = m[i][j]
    return m2

def lp(p, z1=None, z2=None):
    if z1 is None or z2 is None:
        return lambda z1,z2: (abs(z1.real - z2.real)**p + abs(z1.imag - z2.imag)**p)**(1.0/p)
    else:
        return  (abs(z1.real - z2.real)**p + abs(z1.imag - z2.imag)**p)**(1.0/p)

def mkdir():
    try:
        os.makedirs(dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST or not os.path.isdir(dir):
            raise

def argcmp(z1, z2):
    arg1 = math.atan2(z1.imag, z1.real)
    arg2 = math.atan2(z2.imag, z2.real)
    return int(math.copysign(1, arg1 + 2*pi*(arg1 < 0) - arg2 - 2*pi*(arg2 < 0)))

def writegif(filename, images, duration=0.01):
    images2gif.writeGif(dir+filename+".gif", images, duration)
    print "saved", filename+".gif", "to", dir, "at", time.asctime()

def frange(start, stop=None, step=1.0):
    if stop is None:
        stop = start
        start = 0
    while start < stop:
        yield float(start)
        start += step

#--------------------------------------------------------#
#                    Helper Functions                    #
#--------------------------------------------------------#





#--------------------------------------------------------#
#               Image-producing Functions                #
#--------------------------------------------------------#

def funcgraph(funclist, vw=viewwindow(), colorlist=None, auto=False, filename="plot"+str(int(time.time())), gif=False, add=True, avg=False, initcolor=rgb(), save=True):
    if gif:
        images = []
    if auto:
        vw.autograph(funclist)
    data = init(vw, initcolor)
    if colorlist is None:
        colorlist=colors(len(funclist))
    for j in range(len(funclist)):
        f = funclist[j]
        for i in range(vw.dimx-1):
            if colorlist == "rainbow":
                color = rbow(2*pi*i/vw.dimx, 2)
            else:
                color = colorlist[j]
            data.line((vw.xpxcart(i), f(vw.xpxcart(i))), (vw.xpxcart(i+1), f(vw.xpxcart(i+1))), color, add, avg)
            if gif:
                images.append(data.save("", False))
    if gif:
        writegif(filename, images, 0.01)
    return data.save(filename, save)

def polar(rlist ,vw=viewwindow(), colorlist=None, auto=False, gif=False, add=True, avg=False, initcolor=rgb(), filename="polar"+str(int(time.time())), save=True):
    return param([lambda t: r(t)*cos(t) for r in rlist], [lambda t: r(t)*sin(t) for r in rlist], vw, colorlist, auto, gif, add, avg, initcolor, filename, save)

def param(xlist, ylist, vw=viewwindow(), colorlist=None, auto=False, gif=False, add=True, avg=False, initcolor=rgb(), filename="param"+str(int(time.time())), save=True):
    if auto:
        vw.autoparam(xlist,ylist)
    data = init(vw, initcolor)
    if gif:
        images = []
    if colorlist is None:
        colorlist = colors(len(xlist))
    for i in range(len(xlist)):
        x = xlist[i]
        y = ylist[i]
        t = vw.tmin
        if colorlist != 'rainbow':
            color = colorlist[i]
        while t <= vw.tmax:
            if colorlist == 'rainbow':
                color = rbow(vw.tmax - t, 2)
            u = t + vw.tstep
            data.line((x(t), y(t)), (x(u), y(u)), color, add, avg)
            t = u
            if gif:
                images.append(data.save("", False))
    if gif:
        writegif(filename, images, 0.01)
    return data.save(filename, save)

def polarshade(r1, r2, vw=viewwindow(), color=None, auto=False, initcolor=0, filename="polarshade"+str(int(time.time())), save=True):
    if auto:
        vw.autoparam([lambda t: r1(t)*cos(t), lambda t: r2(t)*cos(t)], [lambda t: r1(t)*sin(t), lambda t: r2(t)*sin(t)])
    data = init(vw, initcolor)
    th = vw.thick*(vw.xmax-vw.xmin)/vw.dimx
    if color is None:
        c = rgb(255, 0, 0)
    else:
        c = color
    for i in range(vw.dimx):
        x = vw.xpxcart(i)
        for j in range(vw.dimy):
            y = vw.ypxcart(j)
            t = math.atan2(y, x)
            r = math.sqrt(x**2 + y**2)
            if color == 'rainbow':
                c = rbow(t)
            if r1(t)-th <= r <= r2(t)+th or r2(t)-th <= r <= r1(t)+th:
                data.putpixel((i, j), c)
    return data.save(filename, save)

def basin(f, rootlist, colorlist=None, vw=viewwindow(), df=None, n=10, dots=True, filename="basin"+str(int(time.time())), save=True):
    if df is None:
        df = diffquo(f, .0001)
    if colorlist is None:
        colorlist = colors(len(rootlist))
    data = init(vw)
    for i in range(vw.dimx):
        for j in range(vw.dimy):
            data.putpixel((i, j), colorlist[closest(rootlist, newton(f, vw.xpxcart(i) + vw.ypxcart(j)*1j, df, n))])
    if dots:
        for r in rootlist:
            data.putpoint((r.real, r.imag), 0)
    return data.save(filename, save)

def basin2(f, rootlist, colorlist=None, vw=viewwindow(), df=None, n=10, dist=.3, dots=True, filename="basin2"+str(int(time.time())), save=True):
    if df is None:
        df = diffquo(f, .0001)
    if colorlist is None:
        colorlist = colors(len(rootlist))
    data = init(vw)
    for i in range(vw.dimx):
        for j in range(vw.dimy):
            z = vw.xpxcart(i) + vw.ypxcart(j)*1j
            for k in range(n):
                z = newton(f, z, df, 1)
                index = closest(rootlist, z)
                d = abs(z - rootlist[index])
                if d < dist or k == n-1:
                    data.putpixel((i,j), colorscale(colorlist[index], max(.2, 1 - float(k)/n)))
                    break
    if dots:
        for r in rootlist:
            data.putpoint((r.real,r.imag),0)
    return data.save(filename, save)

def basin3(f, rootlist, colorlist=None, vw=viewwindow(), df=None, n=10, dots=True, filename="basin"+str(int(time.time())), save=True):
    if df is None:
        df = diffquo(f, .0001)
    if colorlist is None:
        colorlist = colors(len(rootlist))
    data = init(vw)
    for i in range(vw.dimx):
        for j in range(vw.dimy):
            z = vw.xpxcart(i) + vw.ypxcart(j)*1j
            for k in range(n + 1):
                data.putpixel((i, j), colorlist[closest(rootlist, z)], add = True, avg = True)
                if k != n:
                    z = newton(f, z, df, 1)
    if dots:
        for r in rootlist:
            data.putpoint((r.real, r.imag), 0)
    return data.save(filename, save)

def newtondist(f, vw=viewwindow(), df=None, n=1, pxdist=1, filename="newtondist"+str(int(time.time()))):
    images = []
    if df is None:
        df = diffquo(f, .0001)
    col = colors(vw.dimx*vw.dimy)
    for k in range(6):
        data = init(vw,0x000000)
        for i in range(vw.dimx):
            for j in range(vw.dimy):
                if i % pxdist == 0 and j % pxdist == 0:
                    z0 = vw.xpxcart(i)+1j*vw.ypxcart(j)
                    z = newton(f,z0,df,n, 1e-1)
                    z0 = newton(f,z0,df,n-1, 1e-1)
                    data.putpoint(vw,((k*z.real + (5-k)*z0.real)/5, (k*z.imag+(5-k)*z0.imag)/5), col[i + j*vw.dimx])
        images.append(data.save("", False))
    writegif(filename, images, 1)
    return data.save(filename, save)

def newtondiff(f, vw=viewwindow(), df=None, n=1, pxdist=1, filename="newtondiff"+str(int(time.time())), save=True):
    if df is None:
        df = diffquo(f, .0001)
    data = init(vw, 0x000000)
    for i in range(vw.dimx):
        for j in range(vw.dimy):
            if (i%pxdist == 0 and j%pxdist == 0):
                z0 = vw.xpxcart(i) + vw.ypxcart(j)*1j
                z = newton(f,z0,df,n, 1e-1)
                if abs(z) < abs(z0):
                    color = rgb(0, 255, 0)
                else:
                    color = rgb(255, 0, 0)
                data.putpixel((i, j), color)
    return data.save(filename, save)

def rk4(f, initlist, vw=viewwindow(), h=.001, colorlist=None, sf=False, sfspread=20, sfcolor=0x0000FF, filename="rk4"+str(int(time.time())), save=True):
    data = init(vw, 0x000000)
    if sf:
        data.data = list(slopefield(f, vw, sfspread, sfcolor, save=False).getdata())
    if colorlist is None:
        colorlist = colors(len(initlist))
    for i in range(len(initlist)):
        c = colorlist[i]
        for h1 in [h, -h]:
            x0, y0 = initlist[i]
            while vw.xmin < x0 < vw.xmax:
                if colorlist == "rainbow":
                    c = rbow(2*pi*(x0-vw.xmin)/(vw.xmax-vw.xmin), 2)
                try:
                    k1 = f(x0, y0)
                    k2 = f(x0 + h1/2, y0 + h1*k1/2)
                    k3 = f(x0 + h1/2, y0 + h1*k2/2)
                    k4 = f(x0 + h1, y0 + h1*k3)
                    x1, y1 = x0 + h1, y0 + (h1/6)*(k1 + 2*k2 + 2*k3 + k4)
                    data.line((x0, y0), (x1, y1), c)
                    x0, y0 = x1, y1
                except:
                    break
    return data.save(filename, save)

def slopefield(f, vw=viewwindow(), spread=20, color=0x0000FF, filename="slopefield"+str(int(time.time())), save=True):
    tmp = vw.thick
    vw.thick = 0
    data = init(vw, 0x000000)
    cartspreadx = vw.xpxcart(spread)-vw.xpxcart(0)
    cartspready = vw.ypxcart(spread)-vw.ypxcart(0)
    for i in range(vw.dimx):
        for j in range(vw.dimy):
            if i % spread == 0 and j % spread == 0:
                theta = math.atan(f(vw.xpxcart(i),vw.ypxcart(j)))
                data.line((vw.xpxcart(i)+cartspreadx*cos(theta)/4,vw.ypxcart(j)-cartspready*sin(theta)/4),
                          (vw.xpxcart(i)-cartspreadx*cos(theta)/4,vw.ypxcart(j)+cartspready*sin(theta)/4),color)
    vw.thick = tmp
    return data.save(filename, save)

def implicit(flist, vw=viewwindow(), decay=100, colorlist=None, filename="implicit"+str(int(time.time())), save=True):
    data = init(vw)
    if colorlist is None:
        colorlist = [0x0000FF]*len(flist)
    for k in range(len(flist)):
        f = flist[k]
        color = colorlist[k]
        for i in range(vw.dimx):
            for j in range(vw.dimy):
                if colorlist == "rainbow":
                    color = rbow(vw.xpxcart(i) + vw.ypxcart(j), 2)
                data.putpixel((i, j), coloradd(data.data[i + j*vw.dimx], colorscale(color, math.e**(-decay*abs(f((vw.xpxcart(i),vw.ypxcart(j))))))))
    return data.save(filename, save)

def implicit2(flist, vw=viewwindow(), decay=100, colorlist=None, filename="implicit2"+str(int(time.time())), save=True):
    data = init(vw)
    if colorlist is None:
        colorlist = [0x0000FF]*len(flist)
    for k in range(len(flist)):
        f = flist[k]
        color = colorlist[k]
        for i in range(vw.dimx):
            for j in range(vw.dimy):
                x = vw.xpxcart(i)
                y = vw.ypxcart(j)
                grad = math.sqrt(gradient(f, (x,y))[0]**2 + gradient(f, (x,y))[1]**2)
                if colorlist == "rainbow":
                    color = rbow(vw.xpxcart(i) + vw.ypxcart(j), 2)
                if abs(f((x, y))) < 10**(-2 + grad/5):
                    data.putpoint((x,y),color)
    return data.save(filename, save)

def levelcurves(f, vw=viewwindow(), decay=100, min=-1, max=1, step=.1, filename="levelcurves"+str(int(time.time())), save=True):
    implicit([(lambda i: lambda (x,y): f((x,y)) - (min + i*step))(i) for i in range(int((max-min)/step)+1)], vw, decay, colors(int((max-min)/step)+1), filename, save)

def color3d(f, vw=viewwindow(), base=1.0, pos=rgb(255,0,0), neg=rgb(0,0,255), filename="color3d"+str(int(time.time())), save=True):
    data = init(vw)
    for i in range(vw.dimx):
        for j in range(vw.dimy):
            (x,y) = (vw.xpxcart(i),vw.ypxcart(j))
            color = 0
            val = f((x,y))
            if val > 0:
                color = colorscale(pos, abs(val/base))
            if val < 0:
                color = colorscale(neg, abs(val/base))
##           color = coloradd(colorscale(pos, .5 + .5*val/base), colorscale(neg, .5 - .5*val/base))
##           color = rbow(pi*(1 + val/base)/2, 1)
##           color = colorscale(rbow(x+y, 2), .5 + .5*val/base)
            data.putpixel((i, j), color)
    return data.save(filename, save)

def zrotate(xlist, ylist, vw=viewwindow(), segs=4, colorlist=None, auto=False, filename=None, gif=False, gifres=1, add=True, avg=False, initcolor=0, save=True):
    if auto:
        vw.autoparam(xlist,ylist)
    if filename is None:
        filename = "zrotate"+str(int(time.time()))
    data = init(vw, initcolor)
    if gif:
        images = []
    if colorlist is None:
        colorlist = [rgb(255, 0, 0)]*len(xlist)
    tlist = [vw.tmin + (vw.tmax-vw.tmin)*j/segs for j in range(segs + 1)]
    while tlist[0] < vw.tmax:
        for (x, y, color) in zip(xlist, ylist, colorlist):
            if colorlist == 'rainbow':
                color = rbow(tlist[0], segs)
            for j in range(len(tlist) - 1):
                data.line((x(tlist[j]), y(tlist[j])), (x(tlist[j+1]), y(tlist[j+1])), color, add, avg)
        if gif:
            images.append(data.save("", False))
        tlist = [t + vw.tstep for t in tlist]
    if gif:
        writegif(filename, images, 0.01)
    return data.save(filename, save)

def gridspanningtree(m, n, d=2, r=0, g="grid", filename=str(int(time.time())), func="spanningtreedfs", gif=False, save=True):
    if gif:
        images = []
    if not isinstance(g, graph):
        (x, y) = (random.randint(0, m - 1), random.randint(0, n - 1))
        g = eval("graph."+g+"(m, n)."+func+"((x,y))")
    vw = viewwindow(dimx = d*m + d - 1, dimy = d*n + d - 1)
    data = init(vw)
    for v1 in g.dict:
        for i in range(-r, r+1):
            for j in range(-r, r+1):
                data.putpixel((d*(v1[0] + 1) - 1 + i, d*(v1[1] + 1) - 1 + j), rgb(255, 0, 0))
    s = {(x,y)}
    q = [(x,y)]
    while q:
        v1 = q.pop(0)
        for v2 in g.dict[v1]:
            if v2 not in s:
                for i in range(1+r, d-r):
                    data.putpixel((d*(v1[0] + 1) - 1 + (v2[0] - v1[0])*i, d*(v1[1] + 1) - 1 + (v2[1] - v1[1])*i), rgb(200, 0, 0))
                q.append(v2)
                s.add(v2)
                if gif:
                    images.append(data.save("", False))
    filename = "gst "+str((m,n))+" "+filename
    ret = data.save(filename, save)
    if gif:
        writegif(filename, images, 0.01)
    return ret

def gridspanningtree2(m, n, d=2, r=0, g="grid", filename=str(int(time.time())), treefunc=None, func="dfs", gif=False, save=True):
    if gif:
        images = []
    if not isinstance(g, graph):
        g = eval("graph."+g+"(m, n)")
    if treefunc is not None:
        (x, y) = (random.randint(0, m - 1), random.randint(0, n - 1))
        g = eval("g."+treefunc+"((x,y))")
    (x, y) = (random.randint(0, m - 1), random.randint(0, n - 1))
    g = eval("g."+func+"((x,y), mode=True)")
    vw = viewwindow(dimx = d*m + d - 1, dimy = d*n + d - 1)
    data = init(vw)
    for p in g:
        try:
            for i in range(-r, r+1):
                for j in range(-r, r+1):
                    data.putpixel((d*(p[0] + 1) - 1 + i, d*(p[1] + 1) - 1 + j), rgb(255, 0, 0))
        except:
            for i in range(-r, r+1):
                for j in range(-r, r+1):
                    data.putpixel((d*(p[1][0] + 1) - 1 + i, d*(p[1][1] + 1) - 1 + j), rgb(255, 0, 0))
            for i in range(1+r, d-r):
                    data.putpixel((d*(p[0][0] + 1) - 1 + (p[1][0] - p[0][0])*i, d*(p[0][1] + 1) - 1 + (p[1][1] - p[0][1])*i), rgb(200, 0, 0))
        if gif:
            images.append(data.save("", False))
    filename = "gst2 "+str((m,n))+" "+filename
    ret = data.save(filename, save)
    if gif:
        writegif(filename, images, 0.01)
    return ret

def graphpict(m, n, g="grid", treefunc="dfs", func="bfs", colorfunc="hilbert", gif=False, gifres=None, v0=None, v1=None, rand=True, permute=0, save=True):
    if gif:
        images = []
    if isinstance(g, graph):
        area = len(g.dict)
    else:
        area = m*n
    if not colorfunc.startswith("rbow"):
        clist = colorcube(colorfunc)
    if func == "hilbert":
        l = hilbert(int(math.ceil(math.log(max(m, n), 2))))
        gstr = "None"
    else:
        if not isinstance(g, graph):
            gstr = g
            g = eval("graph."+g+"(m, n)")
        else:
            gstr = "graph"
        if v0 is None:
            v0 = random.choice(g.dict.keys())
        if treefunc is not None:
            if v1 is None:
                v1 = random.choice(g.dict.keys())
            g = eval("graph.spanningtree(g."+treefunc+", v1, rand)")
        l = eval("g."+func+"(v0, rand)")
    if gif and gifres is None:
        gifres = math.ceil(area/150.0)
    vw = viewwindow(dimx=m, dimy=n)
    data = init(vw)
    i = 0
    for p in l:
        if colorfunc.startswith("rbow"):
            data.putpixel(p, eval(colorfunc+"(2*pi*i/area, 1)"))
        else:
            data.putpixel(p, rgb(*[c*4 for c in clist[min(64**3*i/area, 64**3 - 1)]], permute=permute))
        if gif and (i+1)%gifres == 0:
            images.append(data.save("", False))
        i += 1
    filename = str((m, n))+" "+gstr+" spanningtree"+str(treefunc)+" "+func+" "+colorfunc+" "+str(rand)+" "+str((v0,v1))+" "+str(int(time.time()))
    ret = data.save(filename, save)
    if gif:
        writegif(filename, images, 0.001)
    return ret

def voronoi(rootlist, colorlist=None, vw=viewwindow(), dots=True, metric=lp(2), filename="voronoi"+str(int(time.time())), save=True):
    if colorlist is None:
        colorlist = colors(len(rootlist))
    data = init(vw)
    for i in range(vw.dimx):
        for j in range(vw.dimy):
            data.putpixel((i, j), colorlist[closest(rootlist, vw.xpxcart(i) + vw.ypxcart(j)*1j, metric)])
    if dots:
        for r in rootlist:
            data.putpoint((r.real, r.imag), 0)
    return data.save(filename, save)

def complexroots(deg, vw, color=rgb(5,0,0), epsilon=0.3, coeffs=[-1, 1], filename=None, save=True):
    data = init(vw)
    for i,l in enumerate(itertools.product(coeffs, repeat = deg+1)):
        if i%len(coeffs)**(deg-4) == 0:
             print float(i)/2**(deg+1)
        for root in np.roots(l):
            data.putpoint((root.real, root.imag), color, True)
    if filename is None:
        filename = "complexroots "+str(deg)+" "+str(int(time.time()))
    return data.save(filename, save)

def polynomialbasin(coeffs, vw, auto=False, n=10, dots=True, mode=1, filename=None, save=True):
    p = polynomial(*coeffs)
    rootlist = list(p.roots())
    rootlist.sort(cmp = argcmp)
    if auto:
        vw.autocomplex(*rootlist)
    if filename is None:
        filename = "polynomialbasin "+str(p)+" "+str(n)+" "+str(int(time.time()))
    return eval("basin"+str(mode)*(mode in {2,3})+"(p, rootlist, colors(len(rootlist)), vw=vw, df=p.deriv(), n=n, dots=dots, filename=filename, save=save)")

def mandelbrot(vw=viewwindow(xmin=-2.3, xmax=0.7, ymin=-1.5, ymax=1.5, dimx=700, dimy=700), n=50, reflect=True, color=rgb(255,0,0), filename=None, save=True):
    data = init(vw)
    color0 = color
    if not callable(color):
        color = lambda t: colorscale(color0, t)
    for p in vw.pxiterator():
        c = vw.xpxcart(p[0]) + vw.ypxcart(p[1])*1j
        if c.imag > -0.1 or not reflect:
            i, z0 = 0, 0
            q = (c.real - 0.25)**2 + c.imag**2
            if q*(q + (c.real-0.25)) < c.imag**2/4 or 16*((c.real+1)**2+c.imag**2) < 1:
                data.putpixel(p, color(1.0))
                if reflect:
                    data.putpixel((p[0], vw.ycartpx(-c.imag)), color(1.0))
            else:
                while abs(z0) < 2 and i < n:
                    z0 = z0*z0 + c
                    i += 1
                data.putpixel(p, color(float(i)/n))
                if reflect:
                    data.putpixel((p[0], vw.ycartpx(-c.imag)), color(float(i)/n))
        else:
            break
    if filename is None:
        filename = "mandelbrot "+str(n)+" "+str(int(time.time()))
    return data.save(filename, save)

def julia(f, vw, n=50, Z=2, color=rgb(255,0,0), filename=None, save=True):
    data = init(vw)
    color0 = color
    if not callable(color):
        color = lambda t: colorscale(color0, t)
    for p in vw.pxiterator():
        z0 = vw.xpxcart(p[0]) + vw.ypxcart(p[1])*1j
        z = z0
        i = 0
        while abs(z) < Z and i < n:
            z = f(z, z0)
            i += 1
        data.putpixel(p, color(float(i)/n))
    if filename is None:
        filename = "julia "+str(n)+" "+str(Z)+" "+str(int(time.time()))
    return data.save(filename, save)

def gif(cmdlist, duration=0.01, filename="gif"+str(int(time.time()))):
    images = []
    for cmd in cmdlist:
        images.append(eval(cmd))
    writegif(filename, images, duration)

def juliazoom(f, vw, z0=0, factor=1.5, frames=20, n=50, duration=0.5, Z=2, color=rgb(255,0,0), filename=None):
    images = []
    for i in range(frames):
        vw0 = viewwindow(z0.real - factor**-i, z0.real + factor**-i, z0.imag - factor**-i, z0.imag + factor**-i, dimx=vw.dimx, dimy=vw.dimy)
        images.append(julia(f, vw0, n, Z, color, filename, False))
    if filename is None:
        filename = "julia "+str(n)+" "+str(Z)+" "+str(int(time.time()))
    writegif(filename, images, duration)

def pict2graphpict(img, condition=None, treefunc="dfs", func="bfs", colorfunc="hilbert", gif=False, gifres=None, v0=None, v1=None, rand=True, permute=0, save=True):
    if condition is None:
        condition = lambda x: x == 0
    if isinstance(img, str):
        try:
            im = Image.open(img)
        except:
            im = Image.open(dir+img)
        (m, n) = im.size
        seq = im.getdata()
        mat = [[0 for i in range(n)] for j in range(m)]
        for i in range(len(seq)):
            mat[i%m][i/m] = condition(seq[i])
        g = graph.gridmatrix(mat)
    elif isinstance(img, Image.Image):
        (m, n) = img.size
        seq = img.getdata()
        mat = [[0 for i in range(n)] for j in range(m)]
        for i in range(len(seq)):
            mat[i%m][i/m] = condition(seq[i])
        g = graph.gridmatrix(mat)
    else:
        (m, n) = (len(img), len(img[0]))
        g = graph.gridmatrix(img)
    return graphpict(m, n, g, treefunc, func, colorfunc, gif, gifres, v0, v1, rand, permute, save)

fileextension = '.png'
dir = ""
<<<<<<< Updated upstream
=======
SCREEN = 1

mkdir()
if SCREEN:
    DC = GetDC(0)
>>>>>>> Stashed changes

vw0 = viewwindow(xmin=3, xmax=300, ymin=0, ymax=5, tmin=0, tmax=2*pi, tstep=pi/500, dimx=1000, dimy=1000, thick=2, axisx=1, axisy=1)
vw1 = viewwindow(xmin=-4, xmax=4, ymin=-4, ymax=4, tmin=0, tmax=2*pi, tstep=pi/500, dimx=700, dimy=700, thick=1, axisx=1, axisy=1)
vw2 = viewwindow(xmin=-2.3, xmax=0.7, ymin=-1.5, ymax=1.5, tmin=0, tmax=2*pi, tstep=pi/200, dimx=200, dimy=200, thick=0, axisx=0, axisy=0)
vw3 = viewwindow(xmin=-pi, xmax=pi, ymin=-pi, ymax=pi, tmin=0, tmax=2*pi, tstep=pi/200, dimx=2000, dimy=2000, thick=6, axisx=0, axisy=0)
vw4 = viewwindow(xmin=-1.96, xmax=1.96, ymin=-1.1, ymax=1.1, tmin=0, tmax=2*pi, tstep=pi/200, dimx=1366, dimy=768, thick=1, axisx=0, axisy=0)
vw5 = viewwindow(xmin=-2.3, xmax=0.7, ymin=-1.5, ymax=1.5, tmin=0, tmax=2*pi, tstep=pi/200, dimx=700, dimy=700, thick=0, axisx=0, axisy=0)
vw6 = viewwindow(xmin=-1.5, xmax=1.5, ymin=-1.5, ymax=1.5, tmin=0, tmax=2*pi, tstep=pi/200, dimx=500, dimy=500, thick=1, axisx=0, axisy=0)
vw7 = viewwindow(xmin=-1, xmax=1, ymin=-2, ymax=2, tmin=0, tmax=2*pi, tstep=pi/500, dimx=800, dimy=800, thick=2, axisx=0, axisy=0)
<<<<<<< Updated upstream

if __name__ == "__main__":
    SCREEN = 1
    mkdir()
    if SCREEN:
        DC = GetDC(0)

    print "Execution began "+time.asctime()
    pict2graphpict("yuanxiang.png", gif=True)
    rk4(lambda x,y: x+y, [(0,1)], vw1, sf=True)
    graphpict(1366, 768, g="grid", treefunc=None, func="esco", gif=True)
    gridspanningtree2(20, 20, d=4, r=0, g="grid", treefunc="spanningtreedfs", func="dfs", gif=True)
=======
#print "Execution began "+time.asctime()
#pict2graphpict("yuanxiang.png", gif=True)
#rk4(lambda x,y: x+y, [(0,1)], vw1, sf=True)
#graphpict(1366, 768, g="grid", treefunc=None, func="esco", gif=True)#, v1=(0,0), v0=(0,0))
#gridspanningtree2(20, 20, d=4, r=0, g="grid", treefunc="spanningtreedfs", func="dfs", gif=True)
>>>>>>> Stashed changes
