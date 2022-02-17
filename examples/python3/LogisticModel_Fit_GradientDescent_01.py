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
numP = 50
numN = 50

# Load data
data = numpy.loadtxt( open( sys.argv[ 1 ], 'rb' ), delimiter = ',' )
P = data[ data[ : , 2 ] == 1 ]
N = data[ data[ : , 2 ] == 0 ]
numpy.random.shuffle( P )
numpy.random.shuffle( N )
data = numpy.concatenate( ( P[ : numP , : ], N[ : numN , : ] ), axis = 0 )
numpy.random.shuffle( data )

# Configure model
m = PUJ.Model.Logistic( )
m.setParameters( [ 1 for i in range( data.shape[ 1 ] ) ] )
print( 'Initial model = ' + str( m ) )

# Configure cost
J = PUJ.Model.Logistic.Cost( m, data[ : , 0 : -1 ], data[ : , -1 : ] )

# Debugger
debugger = PUJ.Optimizer.Debug.Simple
##debugger = PUJ.Optimizer.Debug.PlotPolynomialCost( data[ : , 0 : -1 ], data[ : , -1 : ] )

# Fit using an optimization algorithm
opt = PUJ.Optimizer.GradientDescent( J )
opt.setDebugFunction( debugger )
opt.setLearningRate( 1e-4 )
opt.setNumberOfIterations( 10000 )
opt.setNumberOfDebugIterations( 10 )
opt.Fit( )

# Show results
print( '===========================================' )
print( '= Iterations       :', opt.realIterations( ) )
print( '= Fitted model     :', m )
print( '===========================================' )

## eof - $RCSfile$
