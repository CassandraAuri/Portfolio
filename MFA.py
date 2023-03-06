import numpy as np


def MFA(X, meanX, U):  # Transforms X, meanX, and the position vector of the sallite into vectors MFA and transforms V into MFA
    Xmfa = np.zeros((3, len(X)))
    bmean = np.sqrt(meanX[0]**2+meanX[1]**2+meanX[2]**2)
    xmfa_unitary2 = bmean/meanX
    Xmfa[2] = np.dot(X, np.array(xmfa_unitary2).T)
    Xazim = np.cross(np.array(xmfa_unitary2), U)
    bazim = np.sqrt(Xazim[0]**2+Xazim[1]**2+Xazim[2]**2)
    xmfa_unitary1 = Xazim/bazim
    Xmfa[1] = np.dot(X, np.array(xmfa_unitary1).T)
    xpol = np.cross(xmfa_unitary1, xmfa_unitary2)
    bxpol = np.sqrt(xpol[0]**2+xpol[0]**2+xpol[0]**2)
    xmfa_unitary0 = xpol/bxpol
    Xmfa[0] = np.dot(X, np.array(xmfa_unitary0).T)
    return Xmfa
