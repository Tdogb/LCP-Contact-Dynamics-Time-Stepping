import lemkelcp as lcp
import numpy as np

def main():
    M = np.array([[2,1],
            [0,2]])
    q = np.array([-1,-2])

    sol = lcp.lemkelcp(M,q)
    z,exit_code,exit_string = sol
    print(z)

    print("----------")

if __name__ == "__main__":
    main()