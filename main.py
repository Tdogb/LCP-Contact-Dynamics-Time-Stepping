import numpy as np
import lemkelcp as lcp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def main():
    #Constants
    q = np.matrix([0,0,0]).T
    l = 0.5
    m = 1
    r = 0.05
    J = 0.002
    mu = 0.6
    M = np.diag([m,m,J])
    h = 0.01

    #Initial Conditions
    theta0 = np.radians(30)

    #Descision Variables
    cn = 0
    lamb = 0
    beta = np.matrix([0,0,0,0]).T

    #Variables
    is_static = True

    ql = np.matrix([0,2,0]).T
    vl = np.matrix([0,0,4]).T
    qlp1 = np.matrix([0,0,0]).T
    vlp1 = np.matrix([0,0,0]).T

    k = np.matrix([0,-9.8*m,0]).T
    D = np.matrix([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0]]).T
    e = np.matrix([1,1,1,1]).T
    n = np.matrix([0,1,0]).T
    alpha0 = 0

    Mlcp = np.zeros((6,6))
    Mlcp[5] = np.array([mu,-1,-1,-1,-1,0])
    Mlcp[:,5] = np.array([0,1,1,1,1,0])

    vels0 = []
    vels1 = []
    vels2 = []
    for i in range(0,100):
        vels0.append(vl[0,0])
        vels1.append(vl[1,0])
        vels2.append(vl[2,0])
        print("vels")
        vlp1 = np.linalg.inv(M) * (h*(n*cn + D*beta + k)) + vl
        qlp1 = ql + vl*h
        row1 = np.ndarray.flatten(np.array([n.T*qlp1-alpha0]))
        row2 = np.ndarray.flatten(np.array(D.T*vlp1))
        row3 = np.ndarray.flatten(np.array([0])) #mu*cn - e.T*beta
        # qlcptemp = np.array([np.ndarray.flatten(np.array([n.T*qlp1-alpha0])),[lamb*e+D.T*vlp1],np.ndarray.flatten(np.array([mu*cn - e.T*beta]))])
        qlcpflattened = np.hstack((np.hstack((row1,row2)),row3))
        # qlcp = np.matrix([[n.T*qlp1-alpha0],[lamb*e+D.T*vlp1],[mu*cn - e.T*beta]])

        ql = qlp1
        vl = vlp1
        print(np.array(Mlcp))
        print(np.array(qlcpflattened))
        z,exit_code,exit_string = lcp.lemkelcp(np.array(Mlcp), np.array(qlcpflattened))
        if exit_code != 0:
            print("solution not found!!!!!!!!!!")
            print(i)
            plt.plot(vels0)
            plt.plot(vels1)
            plt.plot(vels2)
            plt.show()
        print("-----sol-----")
        print(z)
        cn = z[0]
        beta = np.matrix(z[1:5]).T
        lamb = z[5]
    plt.plot(vels0)
    plt.plot(vels1)
    plt.plot(vels2)
    plt.show()

if __name__ == "__main__":
    main()

