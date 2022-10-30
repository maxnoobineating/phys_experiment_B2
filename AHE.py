from re import X
import numpy as np
from numpy import array as mat
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from math import *  
import copy


def regress(Bs, Rs):
    A = np.vstack([Bs, np.ones(len(Bs))]).T

    x, residual, _, _ = np.linalg.lstsq(A, Rs, rcond=None)
    m, c = x
    return m, c, residual    
    
# Bs = list()
# Rs = list()
# with open("AHE1.txt", "r") as fh:
#     for line in fh:
#         sp = line.split()
#         Bs.append(float(sp[0]))
#         Rs.append(float(sp[1]))
# data = mat([Bs, Rs])
# print(data)
# plt.plot(data[0], data[1])
# plt.show()
# index = mat(range(round(min(Bs)), round(max(Bs))+1))
# index = np.arange(min(Bs), max(Bs), (max(Bs)-min(Bs))/10)

def sign(x):
    return x/abs(x)


def cutBranch(data):
    Bs = data[0]
    temp = Bs[0]
    direction = sign(Bs[1] - Bs[0])
    for n, b in enumerate(Bs[1:]):
        if sign(b-temp)!=direction:
            break
        temp = b
    return ([data[0][:n], data[1][:n]], [data[0][n:], data[1][n:]])

fid = 70
def find_linear(Bs, Rs):
    limit = (abs(max(Bs)) + abs(min(Bs)))/2
    vec = np.vstack((Bs, Rs)).T
    res = list()
    for lim in np.arange(min(np.abs(Bs)), limit, limit/fid):
        BR = mat(list(filter(lambda v: abs(v[0])<=lim, vec)))
        m, c, residual = regress((BR.T)[0], (BR.T)[1])
        if not residual:
            continue
        res.append((m, c, residual[0], BR.T))
    return res



class DataPoints:
    
    def __init__(self, **kwarg):
        self.Bs = list()
        self.Rs = list()
        self.file = str()
        if 'fileName' in kwarg:
            self.file = kwarg.get('fileName')            
            # print("initialized via file")
            with open(self.file, "r") as fh:
                for line in fh:
                    sp = line.split()
                    self.Bs.append(float(sp[0]))
                    self.Rs.append(float(sp[1]))
        elif 'data_points' in kwarg:
            dP = kwarg.get('data_points', None)
            # print("initialized via data_points")
            self.Bs = copy.deepcopy(dP.Bs)
            self.Rs = copy.deepcopy(dP.Rs)
        elif 'raw_data' in kwarg:   
            rawD = kwarg.get('raw_data', None)         
            # print("initialized via raw_data")
            self.Bs = mat(rawD[0])
            self.Rs = mat(rawD[1])
        else:
            print("initialization failed!")
            
    
    def save(self, fileName=''):
        if fileName == '':
            if self.file == '':
                fileName = 'tempData.txt'
            else:
                fileName = self.file.split('.')[0]+"_mod.txt"
                print(fileName)

        with open(fileName, 'w') as fh:
            fh.write(str(self.Bs)+"\n\n"+str(self.Rs))
        
    def data(self):
        return (self.Bs, self.Rs)


    # deprecated in python 3!
    # def __getslice__(self, n0=0, n1=None):
    #     res = DataPoints(data_points=self)
    #     res.Bs = res.Bs[n0:n1]
    #     res.Rs = res.Rs[n0:n1]
    #     return res
    
    def __getitem__(self, k):
        if isinstance(k, slice):
            res = self.copy()
            res.Bs = res.Bs[k]
            res.Rs = res.Rs[k]
            return res
        else:
            res = DataPoints(raw_data=[[self.Bs[k]], [self.Rs[k]]])
            return res
        # can be integrated but fuck it
        
    def copyCut(self, n0, n1):
        res = self.copy()
        res.Bs = res.Bs[n0:n1]
        res.Rs = res.Rs[n0:n1]
        return res
    
    # return 2 branches of DataPoints of B-swip (left, right)
    def copyCutBranch(self):
        temp = self.Bs[0]
        direction = sign(self.Bs[1] - self.Bs[0])
        for n, b in enumerate(self.Bs[1:]):
            if sign(b-temp)!=direction:
                break
            temp = b
        branch1 = self.copyCut(0, n)
        branch2 = self.copyCut(n, None)
        
        return (branch1, branch2)
    
    def copy(self):
        return DataPoints(data_points=self)
    
    # can you make DataPoint a sequence class, so that these operation can be done directly on dP?
    def find_linear_main(self, fid=70):
        limit = (abs(max(self.Bs)) + abs(min(self.Bs)))/2
        vec = np.vstack((self.Bs, self.Rs)).T
        res = list()
        for lim in np.arange(min(np.abs(self.Bs)), limit, limit/fid):
            dP = DataPoints(raw_data = mat(list(filter(lambda v: abs(v[0])<=lim, vec))).T)
            m, c, residual = regress(dP.Bs, dP.Rs)
            if not residual:
                continue
            res.append((m, c, residual[0], dP))
        return res

    def regress(self):
        m, c, residual = regress(self.Bs, self.Rs)
        return (m, c, residual[0])
         
    def plot(self):
        plt.plot(self.Bs, self.Rs, '.k')
        plt.show()
        
    def index(self, fid=10):
        # ind = mat(range(round(min(Bs)), round(max(Bs))+1))
        ind = np.arange(min(self.Bs), max(self.Bs), (max(self.Bs)-min(self.Bs))/fid)
        return ind
    
    def extend(self, dP):
        self.Bs = mat(list(self.Bs) + list(dP.Bs))
        self.Rs = mat(list(self.Rs) + list(dP.Rs))
        
    def __add__(self, other):
        if type(other)==type(tuple()):
            x, y = other
            return DataPoints(raw_data=(self.Bs + x, self.Rs + y))
        elif type(other)==type(self):
            res = self.copy()
            res.extend(other)
            return res
    
    def __len__(self):
        return len(self.Bs)
    
    def __str__(self):
        return "Bs:\n"+str(self.Bs)+"\n\nRs:\n"+str(self.Rs)

# AHE = (I, dI, V, dV, dB)
AHE1 = (1.00, 5.0E-4, 0.171449, 1.0E-5, 1.0E-3)
AHE2_1 = (1.00, 5.0E-4, 0.033, 1.0E-5, 1.0E-3)
AHE2_2 = (1.00, 5.0E-4, 0.033, 1.0E-5, 1.0E-3)
def main_regress(AHE, file="AHE1.txt"):
    dP = DataPoints(fileName=file)
    dP_up, dP_down = dP.copyCutBranch()
    
    lin_ups = dP_up.find_linear_main(fid=100)
    lin_downs = dP_down.find_linear_main(fid=100)
    
    lin_up = regress_pruning(lin_ups, AHE)[-1]
    lin_down = regress_pruning(lin_downs, AHE)[-1]
    
    cup = lin_up[1]
    cdown = lin_down[1]
    cavg = (cup+cdown)/2
    # print("cavg: ", cavg)
    # print(lin_up[-1])
    
    dcup = cavg - cup
    dcdown = cavg - cdown
    
    dP2 = (lin_up[-1]+(0, dcup)) + (lin_down[-1]+(0, dcdown))
    
    # m, c, resid = dP2.regress()
    # print(dP2)
    # print((m, c, resid))
    
    return dP2

def m_error_prop(AHE, dP):
    m, _, _ = dP.regress()
    dR = R_error_prop(*AHE, m=m)
    dB = AHE[-1]
    sum = 0
    for i in range(len(dP.Bs)):
        dP.Bs[i] += dB
        m_dmp, _, _ = dP.regress()
        dP.Bs[i] -= dB*2
        m_dmm, _, _ = dP.regress()
        dP.Bs[i] += dB
        sum += ((m_dmp-m_dmm)/2)**2
        
    for i in range(len(dP.Rs)):
        dP.Rs[i] += dR
        m_dmp, _, _ = dP.regress()
        dP.Rs[i] -= dR*2
        m_dmm, _, _ = dP.regress()
        dP.Rs[i] += dR
        sum += ((m_dmp-m_dmm)/2)**2
    sum = sqrt(sum)
    return sum
    

e_electron = 1.60217663E-19
mu_permeability = 600
rho_resistivity = 6.93E-8
ratio_gauss = 1E4
def carrierDensity_error_prop(d, m, d_err, m_err):
    args = (e_electron, mu_permeability, d, m)
    errors = (0, 1, d_err, m_err)
    return error_prop(carrierDensity_inner, args, errors)

def carrierDensity_inner(e, mu, d, m):
    return mu/(m*ratio_gauss*d*e)

def carrierDensity(d, m):
    return carrierDensity_inner(e_electron, mu_permeability, d, m)
    # return mu_permeability/(m*ratio_gauss*d*e_electron)
    
def mobility(d, m):
    return m*ratio_gauss*d/(mu_permeability*rho_resistivity)

def hallCoeff(d, m):
    return m*d*ratio_gauss/mu_permeability
    
def error_prop(func, args, errors):
    assert(len(args) == len(errors) & len(errors) == func.__code__.co_argcount)
    vars = func.__code__.co_varnames
    arg_dic = {var:arg for var, arg in zip(vars, args)}
    # func_base = func(**arg_dic) -> no use
    sum = 0
    for var, err in zip(vars, errors):
        arg_dic[var] += err
        func_dfunc_p = func(**arg_dic)
        arg_dic[var] -= err*2
        func_dfunc_m = func(**arg_dic)
        arg_dic[var] += err
        sum += ((func_dfunc_p - func_dfunc_m)/2)**2
    sum = sqrt(sum)
    return sum
        
    

# find the lin dP within criteria
def regress_pruning(lin, AHE):
    def func(vec):
        m, c, resid, dP = vec
        sig = sqrt(resid/len(dP))
        return sig <= R_error_prop(*AHE, m=m)
    
    return list(filter(func, lin))

def R_error_prop(I, dI, V, dV, dB, m):
    fdI = -V/I**2
    fdV = 1/I
    return sqrt((fdI*dI)**2+(fdV*dV)**2)+dB*m
    

def main_anime():
    dP = DataPoints(fileName="AHE1.txt")
    res = dP.find_linear_main(fid=70)
    for n, ele in enumerate(res):
        print(n, ", ", ele[2], sep='')

    fig, ax = plt.subplots()
    # x = np.arange(0, 2*np.pi, 0.01)
    m0, c0, _, _ = res[0]
    index = dP.index()
    dots, = ax.plot(dP.Bs, dP.Rs)
    line, = ax.plot(index, m0*index+c0)
    

    def animate(i):
        dots.set_xdata(res[i][3].Bs)
        dots.set_ydata(res[i][3].Rs)  # update the data.
        mi, ci, _, _ = res[i]
        line.set_xdata(index)
        line.set_ydata(mi*index + ci)
        return dots, line, 

    ani = animation.FuncAnimation(
        fig, animate, interval=20, blit=True, save_count=10)
    
    plt.show()
    # writer = animation.FFMpegWriter(
    #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("AHE1_regress.mp4") #, writer=writer)

    # print(m)
    # print(c)
    # _ = plt.plot(Bs, Rs, 'o', label='Original data', markersize=1.)
    # _ = plt.plot(index, m*index + c, 'r', label='Fitted line')
    # _ = plt.legend()
    # plt.show()



def show_time_sequence(dP):
    fig, ax = plt.subplots()

    # x = np.arange(0, 2*np.pi, 0.01)
    line, = ax.plot(dP.Bs, dP.Rs, '.')


    def animate(i):
        print("points: ", i)
        input()
        line.set_xdata(dP.Bs[:i+1])
        line.set_ydata(dP.Rs[:i+1])  # update the data.
        return line,

    ani = animation.FuncAnimation(
        fig, animate, interval=20, blit=True, save_count=len(dP.Bs))

    # To save the animation, use e.g.

    # ani.save("movie.mp4")

    # or

    # writer = animation.FFMpegWriter(
    #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("movie.mp4", writer=writer)

    plt.show()



def aniTest():
    fig, ax = plt.subplots()

    x = np.arange(0, 2*np.pi, 0.01)
    line1, = ax.plot(x, np.sin(x))
    line2, = ax.plot(x, np.cos(x))

    def animate(i):
        line1.set_ydata(np.sin(x + i / 50))  # update the data.
        line2.set_ydata(np.cos(x + i / 50))
        return line1, line2, 

    ani = animation.FuncAnimation(
        fig, animate, interval=20, blit=True, save_count=50)

    # To save the animation, use e.g.
    #v
    #
    # or
    #
    # writer = animation.FFMpegWriter(
    #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("movie.mp4", writer=writer)

    plt.show()



# AHE = AHE1
# file="AHE1.txt"
# dP = DataPoints(fileName=file)
# dP_up, dP_down = dP.copyCutBranch()

# lin_ups = dP_up.find_linear_main(fid=100)
# lin_downs = dP_down.find_linear_main(fid=100)

# lin_up = regress_pruning(lin_ups, AHE)[-1]
# lin_down = regress_pruning(lin_downs, AHE)[-1]

# cup = lin_up[1]
# cdown = lin_down[1]
# cavg = (cup+cdown)/2
# print("cavg: ", cavg)
# print(lin_up[-1])

# dcup = cavg - cup
# dcdown = cavg - cdown

# dP2 = (lin_up[-1]+(0, dcup)) + (lin_down[-1]+(0, dcdown))

# m, c, resid = dP2.regress()
# print(dP2)
# print((m, c, resid))