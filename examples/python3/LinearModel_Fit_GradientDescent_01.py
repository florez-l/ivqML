## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, sys
sys.path.insert( 0, '../../lib/python3' )

import PUJ

## ----------
## -- Main --
## ----------

# Load data
data = numpy.loadtxt( open( sys.argv[ 1 ], 'rb' ), delimiter = ',' )

# Configure model
m = PUJ.Model.Linear( )
m.setParameters( [ 0 for i in range( data.shape[ 1 ] ) ] )
print( 'Initial model = ' + str( m ) )

# Configure cost
J = PUJ.Model.Linear.Cost( m, data[ : , 0 : -1 ], data[ : , -1 : ] )

# Analytical model
a = PUJ.Model.Linear( numpy.matrix( data ) )

# Debugger
debugger = PUJ.Optimizer.Debug.Simple

# Fit using an optimization algorithm
opt = PUJ.Optimizer.GradientDescent( J )
opt.setDebugFunction( debugger )
opt.setLearningRate( 1e-6 )
opt.setNumberOfIterations( 100000 )
opt.setNumberOfDebugIterations( 10 )
opt.Fit( )

# Show results
print( '===========================================' )
print( '= Iterations       :', opt.realIterations( ) )
print( '= Fitted model     :', m )
print( '= Analytical model :', a )
print( '===========================================' )

# Keep showing figures
debugger.KeepFigures( )

## eof - $RCSfile$
