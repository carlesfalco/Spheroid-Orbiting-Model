import numpy as np
import numpy.linalg as lin


class spheroid_model:

    def __init__(self, L, tmax, dt, init_positions, init_velocities, init_ecm, Fa, Fa_y):
        
        # Simulation parameters
        
        self.dt = dt
        self.tmax = tmax
        self.sol_x = [init_positions]
        self.sol_v = [init_velocities]
        self.sol_y = [init_ecm]
        self.L = L 
        self.N = len(init_positions)
        self.M = len(init_ecm)
        
        self.dvdt = np.zeros(init_velocities.shape)
        self.dydt = np.zeros(init_ecm.shape)
        
        # Active-drag forces parameter
 
        u = 3
        self.alpha = 0.1
        self.beta = self.alpha/u**2
        
        # Interaction kernel parameters for cells
  
        self.Fr = 100
        self.Fa = Fa
        self.dr = 15
        self.da = self.dr*2.5
        self.s = 1.25
        
        self.Fstar = 32*self.s*(self.da-self.dr)*(3*self.da**2+4*self.dr*self.da + 3*self.dr**2)/5/(11*self.s + 6)/self.dr**3
        
        if self.Fr <= self.Fstar*self.Fa:
            print(self.Fr,self.Fstar*self.Fa)
            print('Potential for cell-cell interaction not H-stable')
        
        # Interaction kernel parameters for ecm
 
        self.Fr_y = self.Fr
        self.Fa_y = Fa_y
        self.dr_y = self.dr
        self.da_y = self.da
        self.s_y = self.s
        
        self.thetamax = np.pi/3
        
        self.Fstar_y = 32*self.s_y*(self.da_y-self.dr_y)*(3*self.da_y**2+4*self.dr_y*self.da_y + 3*self.dr_y**2)/5/(11*self.s_y + 6)/self.dr_y**3
        
        if self.Fr_y <= self.Fstar_y*self.Fa_y:
            print(self.Fr_y, self.Fstar_y*self.Fa_y)
            print('Potential for cell-ecm interaction not H-stable')
    
    def diffW(self,r): # cell-cell interaction kernel (repulsion + adhesion)
        return self.Fr*(self.dr/2)**(3-self.s*2)*(r+1e-30)**(2*self.s - 3)*(r > 0)*(r < self.dr/2) - 2*self.Fr*(r - self.dr)/self.dr*(r >= self.dr/2)*(r <= self.dr) +4*self.Fa*(r - self.da)*(r - self.dr)/(self.da - self.dr)**2*(r > self.dr)*(r <= self.da)
    
    def diffVr(self,r): # cell-matrix repulsion kernel
        return self.Fr_y*(self.dr_y/2)**(3-self.s_y*2)*(r+1e-30)**(2*self.s_y - 3)*(r > 0)*(r < self.dr_y/2) - 2*self.Fr_y*(r - self.dr_y)/self.dr_y*(r >= self.dr_y/2)*(r <= self.dr_y)
    
    def diffVa(self,r): # cell-matrix adhesion kernel
        return 4*self.Fa_y*(r - self.da_y)*(r - self.dr_y)/(self.da_y - self.dr_y)**2*(r > self.dr_y)*(r <= self.da_y)
    
    def grad_radpot_vectorised(self,x, y, diffr): # force calculation from interaction kernel (diffr)
        
        X = x[np.newaxis, :, :] - y[:, np.newaxis, :]
        norms = lin.norm(X, axis=-1, keepdims=True)
        return diffr(norms) * X / (norms + 1e-80)
        
    def angle_check_vectorised(self,x, v, y): # check cell-matrix alignment angle
        XY = y[ :,np.newaxis,:] - x[np.newaxis,:, :] 
        cos_angle = np.sum(XY * v, axis=-1) / (lin.norm(XY,axis = -1) * lin.norm(v, axis=-1))
        return 1 * (cos_angle >= np.cos(self.thetamax))
    
    def dzdt(self,x,v,y):
        
        vmod = self.alpha - self.beta*lin.norm(v,axis = 1)**2
        self.dvdt = np.transpose(np.array([vmod,vmod]))*v
     
        self.dydt = np.zeros(y.shape)
        
        self.dvdt += np.sum(self.grad_radpot_vectorised(x, x, self.diffW), axis = 0)
        self.dvdt += np.sum(self.grad_radpot_vectorised(x, y, self.diffVa)*self.angle_check_vectorised(x, v, y)[:, :, np.newaxis],axis = 0)
        self.dvdt += np.sum(self.grad_radpot_vectorised(x, y, self.diffVr),axis = 0)
    
        # add equations for dydt (ECM dynamics...)
        
        return v,self.dvdt,self.dydt
    
    def RK4_step(self,x,v,y): # Runge-Kutta step
        
         # Step 1
        k1x, k1v, k1y = self.dzdt(x,v,y)

        # Step 2
        x2 = x + self.dt/2*k1x
        v2 = v + self.dt/2*k1v
        y2 = y + self.dt/2*k1y

        k2x, k2v, k2y = self.dzdt(x2,v2,y2) 

        # Step 3
        x3 = x + self.dt/2*k2x
        v3 = v + self.dt/2*k2v
        y3 = y + self.dt/2*k2y

        k3x,k3v,k3y = self.dzdt(x3,v3,y3)
  
        # Step 4
        x4 = x + self.dt*k3x
        v4 = v + self.dt*k3v
        y4 = y + self.dt*k3y
   
        k4x, k4v, k4y = self.dzdt(x4,v4,y4)
        
        # Conclusion
        xstep = x + (self.dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x)
        vstep = v + (self.dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)
        ystep = y + (self.dt / 6) * (k1y + 2 * k2y + 2 * k3y + k4y)
        
        return xstep, vstep, ystep
        
                                              
    def solver(self, method = 'RK4'): # Call solver to generate simulation, method is RK4, Euler not implemented
       
        t = [0]
        xt,vt,yt = self.sol_x[-1].copy(), self.sol_v[-1].copy(), self.sol_y[-1].copy()
        tr = 0
        while t[-1] < self.tmax:
            
            if method == 'RK4':
            
                xt,vt,yt = self.RK4_step(xt,vt,yt)
                
                if t[-1] - tr > .25: # set time to record data
                    
                    print(t[-1],end = '         \r')
                    self.sol_x.append(xt)
                    self.sol_v.append(vt)
                    tr = t[-1]
            
            t.append(t[-1] + self.dt)
