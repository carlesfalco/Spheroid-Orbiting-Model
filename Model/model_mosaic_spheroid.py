import numpy as np
import numpy.linalg as lin

class mosaic_model:

    def __init__(self ,L, tmax, dt, init_positions1, init_velocities1, init_positions2, init_velocities2, init_ecm):
        
        self.dt = dt
        self.tmax = tmax
        self.sol_x1 = [init_positions1]
        self.sol_v1 = [init_velocities1]
        self.sol_x2 = [init_positions2]
        self.sol_v2 = [init_velocities2]
        self.sol_y = [init_ecm]
        self.L = L
        
        self.N1 = len(init_positions1)
        self.N2 = len(init_positions2)
        self.M = len(init_ecm)
       
        u = 3
        self.alpha1 = 0.1
        self.alpha2 = 0.1
        self.beta1 = self.alpha1/u**2
        self.beta2 = self.alpha2/u**2
        
        # Interaction kernel parameters for cells
        
        self.Fr = 100
        Fc = 5.18
        self.Fa1 = .2*Fc
        self.Fa2 = 0
        self.Fa12 = 0
        
        self.dr = 15
        self.da = self.dr*2.5
        self.s = 1.25
        
        self.Fstar = 32*self.s*(self.da-self.dr)*(3*self.da**2+4*self.dr*self.da + 3*self.dr**2)/5/(11*self.s + 6)/self.dr**3
        
        if self.Fr/self.Fa1 <= self.Fstar:
            print('Potential for cell-cell interaction not H-stable')
        
        
        # Interaction kernel parameters for ecm

        self.Fr_y = 100
        self.Fa_y_1 = .2*Fc
        self.Fa_y_2 = .8*Fc
        self.dr_y = self.dr
        self.da_y = self.da
        self.s_y = 1.25
        
        self.thetamax1 = np.pi/3
        self.thetamax2 = np.pi
        
        #self.Fstar_y = 32*self.s_y*(self.da_y-self.dr_y)*(3*self.da_y**2+4*self.dr_y*self.da_y + 3*self.dr_y**2)/5/(11*self.s_y + 6)/self.dr_y**3
        
        #if self.Fr_y/self.Fa_y <= self.Fstar_y:
            #print('Potential for cell-ecm interaction not H-stable')
    
    def diffW1(self,r):
        return self.Fr*(self.dr/2)**(3-self.s*2)*(r+1e-30)**(2*self.s - 3)*(r > 0)*(r < self.dr/2) - 2*self.Fr*(r - self.dr)/self.dr*(r >= self.dr/2)*(r <= self.dr) + 4*self.Fa1*(r - self.da)*(r - self.dr)/(self.da - self.dr)**2*(r > self.dr)*(r <= self.da)

    def diffW2(self,r):
        return self.Fr*(self.dr/2)**(3-self.s*2)*(r+1e-30)**(2*self.s - 3)*(r > 0)*(r < self.dr/2) - 2*self.Fr*(r - self.dr)/self.dr*(r >= self.dr/2)*(r <= self.dr) + 4*self.Fa2*(r - self.da)*(r - self.dr)/(self.da - self.dr)**2*(r > self.dr)*(r <= self.da)

    def diffW12(self,r):
        return self.Fr*(self.dr/2)**(3-self.s*2)*(r+1e-30)**(2*self.s - 3)*(r > 0)*(r < self.dr/2) - 2*self.Fr*(r - self.dr)/self.dr*(r >= self.dr/2)*(r <= self.dr) + 4*self.Fa12*(r - self.da)*(r - self.dr)/(self.da - self.dr)**2*(r > self.dr)*(r <= self.da)

    def diffVr(self,r):
        return self.Fr_y*(self.dr_y/2)**(3-self.s_y*2)*(r+1e-30)**(2*self.s_y - 3)*(r > 0)*(r < self.dr_y/2) - 2*self.Fr_y*(r - self.dr_y)/self.dr_y*(r >= self.dr_y/2)*(r <= self.dr_y)
    
    def diffVa1(self,r):
        return 4*self.Fa_y_1*(r - self.da_y)*(r - self.dr_y)/(self.da_y - self.dr_y)**2*(r > self.dr_y)*(r <= self.da_y)
    
    def diffVa2(self,r):
        return 4*self.Fa_y_2*(r - self.da_y)*(r - self.dr_y)/(self.da_y - self.dr_y)**2*(r > self.dr_y)*(r <= self.da_y)
   
    def grad_radpot_vectorised(self,x, y, diffr):
        X = x[np.newaxis, :, :] - y[:, np.newaxis, :]
        norms = lin.norm(X, axis=-1, keepdims=True)
        return diffr(norms) * X / (norms + 1e-80)
    
    def forces_ecm(self,x,v,y):
        mat_forces_xy = self.grad_radpot_vectorised(x, y, self.diffV)*self.angle_check_vectorised(x, v, y)[:, :, np.newaxis]
        return self.gamma*np.sum(np.transpose(mat_forces_xy, (1,0,2)), axis = 0) #+ np.sum(self.grad_radpot_vectorised(y, y, self.diffH), axis = 0)
        
    def angle_check_vectorised(self,x, v, y,theta):
        XY = y[ :,np.newaxis,:] - x[np.newaxis,:, :] 
        cos_angle = np.sum(XY * v, axis=-1) / (lin.norm(XY,axis = -1) * lin.norm(v, axis=-1))
        return 1 * (cos_angle >= np.cos(theta))
    
    def dzdt(self,l):
        
        x1,v1,x2,v2,y = l
        
        vmod1 = self.alpha1 - self.beta1*lin.norm(v1,axis = 1)**2
        vmod2 = self.alpha2 - self.beta2*lin.norm(v2,axis = 1)**2
        
        dvdt1 = np.transpose(np.array([vmod1,vmod1]))*v1
        dvdt2 = np.transpose(np.array([vmod2,vmod2]))*v2
     
        dydt = np.zeros(y.shape)
        
        dvdt1 += np.sum(self.grad_radpot_vectorised(x1, x1, self.diffW1), axis = 0)
        dvdt1 += np.sum(self.grad_radpot_vectorised(x1, x2, self.diffW12), axis = 0)
        dvdt1 += np.sum(self.grad_radpot_vectorised(x1, y, self.diffVa1)*self.angle_check_vectorised(x1, v1, y, self.thetamax1)[:, :, np.newaxis],axis = 0)
        dvdt1 += np.sum(self.grad_radpot_vectorised(x1, y, self.diffVr),axis = 0)
        
        dvdt2 += np.sum(self.grad_radpot_vectorised(x2, x2, self.diffW2), axis = 0)
        dvdt2 += np.sum(self.grad_radpot_vectorised(x2, x1, self.diffW12), axis = 0)
        dvdt2 += np.sum(self.grad_radpot_vectorised(x2, y, self.diffVa2)*self.angle_check_vectorised(x2, v2, y, self.thetamax2)[:, :, np.newaxis],axis = 0)
        dvdt2 += np.sum(self.grad_radpot_vectorised(x2, y, self.diffVr),axis = 0)
        
        return [v1,dvdt1,v2,dvdt2,dydt]
    
    def RK4_step(self,l):
        
        # Step 1
        k1 = self.dzdt(l)

        # Step 2
        l2 = [l[i] + self.dt/2 * k1[i] for i in range(len(l))]
        k2 = self.dzdt(l2)

        # Step 3
        l3 = [l[i] + self.dt/2 * k2[i] for i in range(len(l))]
        k3 = self.dzdt(l3)

        # Step 4
        l4 = [l[i] + self.dt * k3[i] for i in range(len(l))]
        k4 = self.dzdt(l4)

        # Conclusion
        lstep = [l[i] + (self.dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(len(l))]

        return lstep

                                              
    def solver(self, method = 'RK4'):
       
        t = [0]
        tr = 0
        x1,v1,x2,v2,y = self.sol_x1[-1], self.sol_v1[-1], self.sol_x2[-1], self.sol_v2[-1],self.sol_y[-1]
        while t[-1] < self.tmax:

            if method == 'RK4':
                
                x1,v1,x2,v2,y = self.RK4_step([x1,v1,x2,v2,y])
                    
                if t[-1] - tr >  .25:
                    
                    print(t[-1],end = '         \r')
                    self.sol_x1.append(x1)
                    self.sol_v1.append(v1)
                    self.sol_x2.append(x2)
                    self.sol_v2.append(v2)
                    self.sol_y.append(y)
                    tr = t[-1]
                    
            t.append(t[-1] + self.dt)