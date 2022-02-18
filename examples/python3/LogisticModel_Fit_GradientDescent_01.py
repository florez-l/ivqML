## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, sys
sys.path.insert( 0, '../../lib/python3' )

import PUJ

## ----------
## -- Main --
## ----------

# Some parameters
numP = 1000
numN = 1000

# Load data
data = numpy.loadtxt( open( sys.argv[ 1 ], 'rb' ), delimiter = ',' )
P = data[ data[ : , 2 ] == 1 ]
N = data[ data[ : , 2 ] == 0 ]
numpy.random.shuffle( P )
numpy.random.shuffle( N )
data = numpy.concatenate( ( P[ : numP , : ], N[ : numN , : ] ), axis = 0 )
numpy.random.shuffle( data )

#meanV = data[ : , 0 : -1 ].mean( axis = 0 )
#C = data[ : , 0 : -1 ] - meanV
#S = numpy.linalg.inv( ( C.T @ C ) / float( data.shape[ 0 ] - 1 ) )
#data[ : , 0 : -1 ] = ( S @ C.T ).T

# Configure model
m = PUJ.Model.Logistic( )
m.setParameters( [ 0 for i in range( data.shape[ 1 ] ) ] )
print( 'Initial model = ' + str( m ) )

# Configure cost
J = PUJ.Model.Logistic.Cost( m, data[ : , 0 : -1 ], data[ : , -1 : ] )

# Debugger
debugger = PUJ.Optimizer.Debug.Simple
##debugger = PUJ.Optimizer.Debug.PlotPolynomialCost( data[ : , 0 : -1 ], data[ : , -1 : ] )

# Fit using an optimization algorithm
opt = PUJ.Optimizer.GradientDescent( J )
opt.setDebugFunction( debugger )
opt.setLearningRate( 1e-3 )
opt.setNumberOfIterations( 10000 )
opt.setNumberOfDebugIterations( 1000 )
opt.Fit( )

# Show results
print( '===========================================' )
print( '= Iterations       :', opt.realIterations( ) )
print( '= Fitted model     :', m )
print( '===========================================' )

y_real = data[ : , -1 : ]
y_est = m.threshold( data[ : , 0 : -1 ] )
K = numpy.zeros( ( 2, 2 ) )

for i in range( y_real.shape[ 0 ] ):
  K[ int( y_real[ i, 0 ] ), int( y_est[ i, 0 ] ) ] += 1
# end for

print( K )


## eof - $RCSfile$
