#Starter code taken from AA274 Homework 1, P1_optimal_control.py
#Data (m, I, l, c) and model taken from https://www.scitepress.org/papers/2014/50186/50186.pdf, then modified

#Note that this code has some oddities given its use of the scipy bvp solver
#Most have to do with variable dimensions, but the code should be fairly well adaptable
#To a different solver. Read the documentation for your solver to set variable dimensions accordingly.

#bvp solver documentation: https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.integrate.solve_bvp.html

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

#These are just for making the animation at the end
import os
import subprocess
import matplotlib.lines as lines
from matplotlib.figure import figaspect as fasp

'Define constants'
lam = 600.0
m = 2.41
g = 9.81
l = 5.22
I = 0.116
c = 0.005
'Define commonly used combinations'
denom = I + m*np.power(l,2)
mgl = m*g*l
ml = m*l

def ode_fun(tau, zz):
    '''
    This function computes the dz given tau and z. It is used in the bvp solver.
    Inputs:
        tau: the independent variable. This MUST be the first argument.
        This is a vector of size (m,) where m is the number of time-steps you take
        In our case, m = 50. See "x" in main() for more details
        
        zz: the time-state matrix. This is of size (n,m) where n is the number of variables you have.
        bvp_solver needs ode_fun to solve for all variables at all time-steps
        This devolves into writing our ode for one time-step, then looping over all steps
        
    Output:
        dz: the time-state derivative matrix. Returns a 2D numpy array (matrix) of size (n,m).
    '''
    dz = np.zeros((zz.shape))

    for i in range(tau.shape[0]):
        'Taking one column of zz corresponds to the values of our variables at a particular time-step'
        z = zz[:,i]

        '''
        Now, treat z as your state vector, and create your ode as you normally would
        First, do some simple operations to avoid constant retyping.
        For example, the first one takes the sin of theta.
        Recall theta = z_3, which is z[2] because of python 0-indexing.
        '''       
        st = np.sin(z[2]) 
        ct = np.cos(z[2])
        u = 0.5*(z[7]*ml*ct/denom - z[5])

        '''
        Note that this is exactly as shown in the notes. You have a z_9 out in front,
        and your first term is z_2, your second is u (as defined in terms of z), etc
        '''
        dz_col = z[8]*np.array([ z[1], u, z[3], (1.0/denom)*(mgl*st - c*z[3] - ml*ct*u),
                0.0,-z[4], -(z[7]/denom)*(mgl*ct+ ml*st*u),z[7]*c/denom - z[6],
            0.0])

        '''
        Now, since this is only for one time step, we need to put this column into
        its appropriate spot in our overall dz matrix. Since it's the i'th time-step,
        we put it into the i'th column of dz. Then we do the whole thing all over again
        for the next time step, i+1
        '''        
        dz[:,i] = dz_col

    return dz


def bc_fun(za, zb):
    '''
    This function computes boundary conditions. It is used in the bvp solver.
    Inputs:
        za: the state vector at the initial time
        This will correspond to all bc's of the form z_i(0)
        
        zb: the state vector at the final time
        This is all bc's of the form z_i(1)
    Output:
        bca: tuple of boundary conditions at initial time
        bcb: tuple of boundary conditions at final time

    IMPORTANT: All boundary conditions must be written as
    f(za,zb) = 0

    Example: z_2(0) = 5 would be written as za[1] - 5

    Also important: Need to have as many bcs as variables
    '''

    #It's helpful to define initial and final states at the start 
    x0 = [0.0, 0.0, 0.0, 0.0] 
    xf = [10.0, 0.0, 0.0, 0.0]

    #We will first solve for the bonus equation given at the end of the setup
    #Note that we're using all zb's, because zb corresponds to the final time
    st = np.sin(zb[2])
    ct = np.cos(zb[2])
    u = 0.5*(zb[7]*ml*ct/denom - zb[5])

    #Note that these next 2 arrays are just p^T and y_dot as given through the z variables
    p_arr = np.array([zb[4],zb[5],zb[6],zb[7]])
    x_d_arr = np.array([zb[1],u,zb[3],(1.0/denom)*(mgl*st - c*zb[3] - ml*ct*u)])

    #u(1)^2 + p(1)^T*y_dot(1) + lambda = 0
    free_time_bonus_eqn = np.power(u,2) + np.inner(p_arr,x_d_arr) + lam

    #Note za[1]-x0[1] is equivalent to z_2(0) - 0 = 0, aka z_2(0) = 0, and so on
    #The free time equation goes with bcb, since it corresponds to a bc at the end time
    bca = np.array([za[0]-x0[0],za[1]-x0[1],za[2]-x0[2],za[3]-x0[3]])
    bcb = np.array([zb[0]-xf[0],zb[1]-xf[1],zb[2]-xf[2],zb[3]-xf[3],
        free_time_bonus_eqn])

    #Stack conditions horizontally to make an np array of size (n,)
    g = np.hstack((bca, bcb))
    return g

def compute_controls(z):
    '''
    This function computes the controls given the solution struct z. It is used in main().
    Input:
        z: the solution struct outputted by solve_bvp
    Outputs:
        accel: the x-ddot control for the wheel
    '''

    #z.y is the (n,m) output matrix of our solution at all time-steps
    vals = z.y
    
    #cos of theta at all time steps
    ct = np.cos(vals[2,:])

    #use our equation for u, and use numpy's default element-wise multiplication
    #to solve for u at all time-steps simultaneously.
    accel = 0.5*(vals[7,:]*ml*ct/denom - vals[5,:])

    return accel

def main():
    '''
    This function solves the specified bvp problem and returns the corresponding optimal contol sequence
    Outputs:
        z: Optimal solution for all variables (struct defined in scipy bvp_solver documentation)
        accel: optimal x_ddot control sequence (np array of size (m,))
    '''

    'First, we define our mesh, aka how fine the discretization of time is'
    x = np.linspace(0,1,50) #Creates a vector of 50 equally spaced points between 0 and 1, inclusive: [0, 0.0204, 0.0408, ... , 1]

    '''
    Next, we define our initial guess.
    Note that this is mostly random except for the last variable,
    which corresponds to t_f, which I guessed was around 5.0
    '''
    initial_guess = np.array([5.0,1.5,-0.2,0.1,-10.0,-5.0,5.0,10.0,5.0]) 

    '''
    For this bvp solver, you must have one n-length guess for each time-step in x.
    Since we have 50 time-steps and 9 variables, we need to have a (9,50) matrix
    Each column of y[:,i] corresponds to time-step x[i]
    '''
    y = np.tile(initial_guess, (x.size,1)) #create (50,9) matrix where each row is initial_guess
    y = y.T #make it into the right shape (9,50)
    z = solve_bvp(ode_fun, bc_fun, x, y) #call bvp solver
    accel = compute_controls(z) #compute controls from your solution
    return z, accel

if __name__ == '__main__':
    z, accel = main()

    #The last variable we solved for was t_f
    tf = z.y[8,0]

    #Our mesh goes from tau = (0,1). Need to scale by tf
    time_mesh = z.x*tf
    
    unicycle_pos = z.y[0,:]
    unicycle_angle = z.y[2,:]


    '''    
    #Uncomment to see plots

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(time_mesh, unicycle_pos,'k-',linewidth=2)
    plt.grid('on')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis([0, 6, -2, 6])
    plt.title('Optimal Wheel Position')

    plt.subplot(1, 3, 2)
    plt.plot(time_mesh, accel,'k-',linewidth=2)
    plt.grid('on')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.axis([0, 6, -15, 15])
    plt.title('Optimal Wheel Acceleration')
    
    plt.subplot(1, 3, 3)
    plt.plot(time_mesh, unicycle_angle,'k-',linewidth=2)
    plt.grid('on')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis([0, 6, -1, 1])
    plt.title('Optimal Rod Angle')

    plt.show()

    '''
    
    #Make a movie to see how the rod and wheel move over time
    files = []
    pos = z.y[0,:]
    ang = z.y[2,:]
    rod_pos = np.vstack((pos,pos+2*l*np.cos((np.pi/2)-ang),np.ones(pos.size),1+2*l*np.sin((np.pi/2)-ang)))

    w,h = fasp(1.)
    fig = plt.figure(figsize=(w,h))
    ax = fig.add_subplot(111)
    for i in range(50):  # 50 frames
        rp = rod_pos[:,i]
        plt.cla()
        plt.plot(pos[i],1,'ro')
        line = lines.Line2D([rp[0],rp[1]],[rp[2],rp[3]])
        ax.add_line(line)
        plt.axis([-4, 14, 0.0, 18])
        fname = '_tmp%03d.png' % i
        print('Saving frame', fname)
        plt.savefig(fname)
        files.append(fname)

    subprocess.call([
        'ffmpeg', '-framerate', '15', '-i', '_tmp%03d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'video_name.mp4'
    ])
    
    for file_name in files:
        os.remove(file_name)

