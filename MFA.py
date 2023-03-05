import numpy as np


def MFA(X, meanX, U):  # Transforms X, meanX, and the position vector of the sallite into vectors MFA and transforms V into MFA
    xmfa_unitary = np.zeros((len(X), 3, 3))
    Xmfa = np.zeros((len(X), 3))
    print(meanX)
    print(meanX[0]**2)
    print(meanX[1]**2)
    print(meanX[2]**2)
    bmean = np.sqrt(meanX[0]**2+meanX[1]**2+meanX[2]**2)
    print(bmean)
    xmfa_unitary[:, :, 2] = bmean/meanX
    print(xmfa_unitary[:, :, 2])
    print(len(np.squeeze(np.asarray(xmfa_unitary[:, :, 2]))))
    print("ree")
    print(len(X))
    print("what the")
    Xmfa[2] = np.dot(np.squeeze(np.asarray(X)),
                     np.squeeze(np.asarray(xmfa_unitary[:, :, 2])))

    Xazim = np.cross(xmfa_unitary[:, :, 1], U)
    bazim = np.sqrt(Xazim[0])**2+Xazim[1]**2+Xazim[2]**2
    xmfa_unitary[:, :, 1] = Xazim/bazim
    Xmfa[1] = np.dot(np.squeeze(np.asarray(X)),
                     np.squeeze(np.asarray(xmfa_unitary[:, :, 1])))

    xpol = np.cross(xmfa_unitary[:, :, 1], xmfa_unitary[:, :, 2])
    bxpol = np.sqrt(xpol[0])**2+xpol[0]**2+xpol[0]**2
    xmfa_unitary[:, 0] = xpol/bxpol
    Xmfa[0] = np.dot(np.squeeze(np.asarray(X)),
                     np.squeeze(np.asarray(xmfa_unitary[:, 0])))
    return Xmfa
