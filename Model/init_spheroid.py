import numpy as np
import numpy.random as ran
import numpy.linalg as lin
from scipy.interpolate import interp1d

def init_positions_N(N = 130, R = 100, L = 240):

    #Input: number of cells (N), spheroid radius (R), computational domain (L > 2*R)
    #Output: initial cell positions
    
    Rinit = R - 15/2
    
    l = np.sqrt(ran.uniform(0, Rinit**2))
    a = np.pi * ran.uniform(0, 2)
    init_positions = [np.array([L/2+ l * np.cos(a),  L/2 + l * np.sin(a)])]
    
    while len(init_positions) < N:
        
        #print(len(init_positions), end = '\r')
        
        l = np.sqrt(ran.uniform(0, Rinit**2))
        a = np.pi * ran.uniform(0, 2)
        z = np.array([L/2+ l * np.cos(a),  L/2 + l * np.sin(a)])
        dist = [lin.norm(z - y) for y in init_positions]
        
        if min(dist) > 9:
            init_positions.append(z)
            
    return np.array(init_positions)

def init_velocities_N(N = 130):
    
    #Input: N
    #Output: initial cell velocities
    
    return np.array([ran.uniform(-np.sqrt(10),np.sqrt(10),2) for i in range(N)])

def init_ecm_M(Np = 0, rp = 0, kp = 0, M0 = 200, R = 100, L = 240):
    
    # Input: number (Np), height (rp), width (kp) of boundary perturbations, R, L
    # M0: number of ECM molecules when boundary = circle
    # Output: ECM molecules positions
    
    if Np == 0: # boundary = circle
        
        theta = [2*np.pi*i/M0 for i in range(M0)]
        return np.array([[L/2+R*np.cos(theta[i]), L/2+R*np.sin(theta[i])] for i in range(M0)])
    
    else: # boundary perturbations
        
        if Np == 4:
            pert = [np.pi/4,np.pi/4 + np.pi/2,np.pi/4 + np.pi,np.pi/4 + 3*np.pi/2]
        elif Np == 3:
            pert = [np.pi,np.pi/3,2*np.pi/3+np.pi]
        elif Np == 2:
            pert = [np.pi/2*3+np.pi/4,np.pi/2+np.pi/4]
        elif Np == 1:
            pert = [np.pi/2]
            
        ramp = kp/R/2 # perturbation half amplitude in rads
        M2 = 1000

        theta = [2*np.pi*i/M2 for i in range(M2)]
        dtheta = theta[1] - theta[0]
        rad = np.array([R + rp*np.sum([np.exp(-(theta[i] - p)**2/ramp**2) for p in pert]) for i in range(M2)])
        
        init_ecm = np.array([[L/2+rad[i]*np.cos(theta[i]), L/2+rad[i]*np.sin(theta[i])] for i in range(M2)])
        rad_diff = np.gradient(rad,theta)
        rad_int = np.sqrt(rad**2+rad_diff**2)
        
        s = [0]
        for i in range(1,len(theta)):
            s.append(np.sum(rad_int[:i])*dtheta)
            
        fun_x = interp1d(s,init_ecm[:,0])
        fun_y = interp1d(s,init_ecm[:,1])

        sspanlength = np.linspace(0,s[-1]-1e-2,M2)
        ecm_length = np.transpose([fun_x(sspanlength),fun_y(sspanlength)])
        
        length = np.sum([lin.norm(ecm_length[i+1] - ecm_length[i]) for i in range(len(sspanlength) - 1)])

        dens = M0/(2*np.pi*R)

        M = int(dens*length)

        sspan = np.linspace(0,s[-1]-1e-2,M)
        init_ecm = np.transpose([fun_x(sspan),fun_y(sspan)])
        
        return init_ecm

def return_N(Np = 0, rp = 0, kp = 0, R = 100):
    
    # Output: number of cells for boundary perturbations
    
    ap = kp/2/R
    
    rho = 130/np.pi/100**2
    
    #A = np.pi*Ry**2 + Np*3/5*rp*ap/2
    A = np.pi*R**2 + Np*np.sqrt(np.pi)*rp*kp*(1 + .5*rp/np.sqrt(2)/R/2)*.5
    return int(A*rho)

