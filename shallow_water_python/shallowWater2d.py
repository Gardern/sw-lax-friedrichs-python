from mpl_toolkits.mplot3d import axes3d
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

import math as m

######################################################
## - usage from command line:                        #
##     python shallowWater.py --n=1000 --TMAX=0.05   #
##     (where:                                       #
##     n = number of grid cells, in each direction   #
##     TMAX = end time simulation)                   #
## - usage from within ipython:                      #
##     from shallowWater import *                    #
##     shallowWater(100,1.,0.01)                     #
######################################################

# Courant Friedrichs Levy condition (CFL) is a necessary condition for convergence
CFL = .99# set so it fulfilles CFL condition, experiment with it
# gravitational constant
g = 9.80665

## shallow water solver 1 dimension
def shallowWater(n,XMAX,TMAX):
    
	TMAX *= XMAX

	# dx = cell size
	dx = 1.*XMAX/n
	# x = cell center
	x = np.arange(dx/2, XMAX, dx)# set to be cell centers
	y = np.reshape(x, (x.size, -1))

	# initialize height h and momentum hu
	h, hu, hv = initialize(x,y,XMAX)

	# add ghost cells (aka frame the array in zeroes)
	h = addGhostCells(h)
	hu = addGhostCells(hu)
	hv = addGhostCells(hv)
	# plot initial conditions with interactive mode on
	#pl.ion()
	#plotVars(x,h,hu,0)
	fig, mX, mY = init3dPlot(x,y,XMAX,dx) # Our plot
	plot3d(fig,mX,mY,h,n) # Our plot
	# time starts at zero
	tsum=0.
	#saveToFile(h,tsum,n,'result.dat','w')
	#saveToFile(hu,tsum,n,'result.dat','a')
	# loop over time
	while tsum<TMAX:

		# apply boundary conditions
		#h = neumannBoundaryConditions(h)
		#hu = neumannBoundaryConditions(hu)
		#hv = neumannBoundaryConditions(hv)
		h = periodicBoundaryConditions(h)
		hu = periodicBoundaryConditions(hu)
		hv = periodicBoundaryConditions(hv)

		# calculate fluxes at cell interfaces and largest eigenvalue
		fhp,fhup,fhvp,eigfp, = fluxesF(h[1:-1,1:],hu[1:-1,1:],hv[1:-1,1:])
		fhm,fhum,fhvm,eigfm = fluxesF(h[1:-1,:-1],hu[1:-1,:-1],hv[1:-1,:-1])
		ghp,ghup,ghvp,eiggp = fluxesG(h[1:,1:-1],hu[1:,1:-1],hv[1:,1:-1])
		ghm,ghum,ghvm,eiggm = fluxesG(h[:-1,1:-1],hu[:-1,1:-1],hv[:-1,1:-1])
		
		maxeig = max(max(eigfp,eigfm),max(eiggm,eiggp))# maximum of eigp and eigm
		
		# calculate time step according to CFL-condition
		dt = calculateDt(dx,maxeig,tsum,TMAX)
		# advance time
		tsum = tsum+dt
		lambd = 1.*dt/dx
		
		# R = Lax-Friedrichs Flux
		Rhf = LxFfluxF(h, fhp, fhm, lambd) 
		Rhg = LxFfluxG(h, ghp, ghm, lambd)
		Rhuf = LxFfluxF(hu, fhup, fhum, lambd) 
		Rhug = LxFfluxG(hu, ghup, ghum, lambd)
		Rhvf = LxFfluxF(hv, fhvp, fhvm, lambd) 
		Rhvg = LxFfluxG(hv, ghvp, ghvm, lambd)

                h[1:-1,1:-1] -= lambd*(0.5 * ((Rhf[:,:-1]-Rhf[:,1:] ) + (Rhg[:-1,:]-Rhg[1:,:])))# update cell average (tip: depends on Rh and lambda)
		hu[1:-1,1:-1] -= lambd*(0.5 * ((Rhuf[:,:-1]-Rhuf[:,1:] ) + (Rhug[:-1,:]-Rhug[1:,:])))# update cell average (tip: depends on Rh and lambda)
		hv[1:-1,1:-1] -= lambd*(0.5 * ((Rhvf[:,:-1]-Rhvf[:,1:] ) + (Rhvg[:-1,:]- Rhvg[1:,:])))# update cell average (tip: depends on Rh and lambda)

		#plotVars(x,h,hu,tsum)
		plot3d(fig,mX,mY,h,n) # Our plot

	#end while (time loop)
	plot3d(fig,mX,mY,h,n) # Our plot
	#saveToFile(h,tsum,n,'result.dat','a')
	#saveToFile(hu,tsum,n,'result.dat','a')

def plotVars(x,h,hu,time,clear=True):
	time = '{0:.5f}'.format(time)
	pl.figure(1)
	if clear:
		pl.clf()
	pl.subplot(211)
	pl.title('water height h(t='+time+')')
	pl.plot(x,h[1:-1],label='u')
	pl.xlabel('x')
	pl.legend()
	pl.subplot(212)
	pl.title('momentum hu(t='+time+')')
	pl.plot(x,hu[1:-1],label='hu')
	pl.xlabel('x')
	pl.legend()
	pl.draw()

def init3dPlot(x,y,XMAX,dx):
        fig = plt.figure()
        
	xplot = np.hstack([-dx/2, x, XMAX + dx/2])
	yplot = np.vstack([[-dx/2], y, [XMAX + dx/2]])
	mX, mY = np.meshgrid(xplot,yplot)
	
	return fig, mX, mY

def plot3d(fig,mX,mY,h,n):
	#plt.cla();
	plt.clf() # Makes it go faster for some reason
	ax = fig.add_subplot(111, projection='3d')
	
	#if clear:
	   #ax.cla(); # Either this or add subplot as above
	
	ax.plot_wireframe(mX, mY, h, rstride=n/25, cstride=n/25)
	#ax.plot_surface(mX, mY, h, rstride=n/25, cstride=n/25, cmap=cm.YlGnBu_r) # Another plot to use
	ax.view_init(50, -125)
	
	# Enforce the axis range
	ax.set_xlim3d(0,1)
        ax.set_ylim3d(0,1)
        ax.set_zlim3d(0,1.3)
	   
	plt.pause(0.01) # Pause figure, makes it interactive
	# time.sleep(1)

def saveToFile(v,t,n,name,mode):
	# mode 'w' (over)writes, 'a' appends
	v = np.hstack([t,CFL,g,n+2,v]) # +2 because of ghost cells
	ff = open(name, mode+'b')
	v.tofile(ff)
	ff.close()
    
def readFromFile(name,clear=True):
	ff = open(name, 'rb')
	data = np.fromfile(ff)
	print "read print", data.size
	ff.close()
	numberCells = data[3]
	n = numberCells-2 # -2 because of ghost cells
	numberParameters = 4 # t, CFL, g, n
	numberVariables = 2 # h,hu
	print name, data.size
	if data.size%(numberCells+numberParameters)*numberVariables==0:
		numberSavedTimeSteps = data.size/((numberCells+numberParameters)*numberVariables)
		data.shape = (numberVariables*numberSavedTimeSteps,(numberCells+numberParameters))
		# plot the last time step
		pl.ion()
		plotVars((1./2. + np.arange(n))/n, data[-2,numberParameters:], data[-1,numberParameters:], data[-1,0], clear)
	else:
		print 'data file inconsistent'
	return data

def error(name,nameReference):
	data = readFromFile(name,True)
	dataRef = readFromFile(nameReference,False)
	numberParameters = 4 # t, CFL, g, n
	h = data[-2,numberParameters+1:-1]
	hu = data[-1,numberParameters+1:-1]
	h_ref = dataRef[-2,numberParameters+1:-1]
	hu_ref = dataRef[-1,numberParameters+1:-1]

	numberCells = data[0,3]
	if (numberCells != dataRef[0,3]):
		print 'solution and reference solution have different number of grid cells'
	else:
		n = numberCells-2 # -2 because of ghost cells
		time = data[-1,0]
		timeRef = dataRef[-1,0]
	if time==timeRef:
		relErrorh = np.sum(np.abs(h-h_ref))/np.sum(np.abs(h_ref))
		absErrorhu = np.sum(np.abs(h-h_ref))/(1.*n)
		time = '{0:.5f}'.format(time)
		relErrorh = '{0:.5f}'.format(relErrorh*100.)
		absErrorhu = '{0:.5f}'.format(absErrorhu*100.)
		print 'relative L1-error of water height h(t='+time+') is '+relErrorh+'%'
		print 'absolute L1-error of momentum hu(t='+time+') is '+absErrorhu+'%'
	else:
		print 'cannot compare solution at different times'
    
def initialize(x,y,XMAX):
	# Makes some random heights
	c = 0.01
	h = .5 + .5*np.exp(-(2.*(x/XMAX-.6))**2/(2.*c**2))
	h += np.exp(-(2.*(x/XMAX-.4))**2/(2.*c**2))
	
	h2 = np.reshape(h, (h.size, 1))
	h = (h*h2)
        
        # Just start parts of it a bit lower
        for i in range(len(h)):
            for j in range(len(h[i])):
                if h[i][j] == 0.25:
                    h[i][j] = h[i][j] = 0.01
	
	# momentum is zero
	hu = 0.*x*y
	hv = 0.*x*y	
	
	# The closest I came to a gaussian 2D distribution (two peaks) (m.exp( -((x - c)**2 + (y - c)**2) / (2*(r)**2) ))
	# Remove this for the random heights
	c = 0.3 # The center
	r = 0.1 # The radius
	for i in range(0, len(h) / 2):
            for j in range(0, len(h[i])):
                h[i][j] = h[i][j] = m.exp( -((x[j] - c)**2 + (y[i] - c)**2) / (2*r**2) ) * .5   
        c = 0.7
        for i in range(len(h) / 2, len(h)):
            for j in range(0, len(h[i])):
                h[i][j] = h[i][j] = m.exp( -((x[j] - c)**2 + (y[i] - c)**2) / (2*r**2) ) * .75
	
	return h, hu, hv

def addGhostCells(var):
	ghosts = np.zeros((var[0].size, 1)) #ghosts on the sides
	var = np.hstack([ghosts, var, ghosts])
	ghosts = np.zeros(var[0].size) #ghosts on top and bottom
	return np.vstack([ghosts,var,ghosts])

def neumannBoundaryConditions(var):
	var[0] = var[1]
	var[-1] = var[-2]
	for i in var:
		i[0] = i[1]
		i[-1] = i[-2]

	return var # var with neumann boundary conditions

def periodicBoundaryConditions(var):
	var[0] = var[-2]
	var[-1] = var[1]
	for i in var:
		i[0] = i[-2]
		i[-1] = i[1]

	return var # var with periodic boundary conditions

def calculateDt(dx,maxeig,tsum,TMAX):
        r = dx/maxeig
	dt = CFL*r
	if tsum + dt > TMAX:
		dt = TMAX - tsum

	return dt #stepsize dtdt

def fluxesF(h,hu,hv):
	fh,fhu,fhv,lambd1,lambd2 = fluxAndLambdaF(h,hu,hv)
	maxeig = np.maximum(np.amax(np.abs(lambd1)), np.amax(np.abs(lambd2)))
	
	return fh,fhu,fhv,maxeig

def fluxesG(h,hu,hv):
	gh,ghu,ghv,lambd1,lambd2 = fluxAndLambdaG(h,hu,hv)
	maxeig = np.maximum(np.amax(np.abs(lambd1)), np.amax(np.abs(lambd2)))
	
	return gh,ghu,ghv,maxeig

def LxFfluxF(q, fqp, fqm, lambd):
        LxFF = 0.5 * ((fqm + fqp) - 0.5 / lambd * (q[1:-1,:-1] - q[1:-1,1:]))
       
	return LxFF

def LxFfluxG(q, gqo, gqn, lambd):
        LxFG = 0.5 * ((gqo + gqn) - 0.5 / lambd * (q[:-1,1:-1] - q[1:,1:-1]))
        
        return LxFG

def fluxAndLambdaF(h,hu,hv):
        c = np.sqrt(g*h)
        #h = np.maximum(0.01, h)
    
	u = hu/h
	fh = hu
	fhu = hu*u + 0.5*g*h*h
	fhv = hv*u
	lambda1 = u + c
	lambda2 = u - c
	return fh,fhu,fhv,lambda1,lambda2 #fluxes fh, fhu and fhv and eigenvalues lambda1, lambda2

def fluxAndLambdaG(h,hu,hv):
        c = np.sqrt(g*h)
        #h = np.maximum(0.01, h)
    
	v = hv/h
	gh = hv
	ghu = hu*v
	ghv = hv*v + 0.5*g*h*h
	lambda1 = v + c
	lambda2 = v - c
	return gh,ghu,ghv,lambda1,lambda2 #fluxes gh, ghu and ghv and eigenvalues lambda1, lambda2

if __name__ == "__main__":

	from optparse import OptionParser
	usage = "usage: %prog [var=value]"
	p = OptionParser(usage)
	p.add_option("--n", type="int", help="number of points")
	p.add_option("--XMAX", type="float", help="length of domain")
	p.add_option("--TMAX", type="float", help="time of simulation")
	(opts, args) = p.parse_args()
	if opts.n == None:
		n = 50
	else:
		n = opts.n
	if opts.XMAX == None:
		XMAX = 1.0
	else:
		XMAX = opts.XMAX
	if opts.TMAX == None:
		TMAX = 1.2
	else:
		TMAX = opts.TMAX

	shallowWater(n,XMAX,TMAX)
	#error('result.dat', 'result_n1000_TMAX045.dat')
    