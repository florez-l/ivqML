## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, sys
sys.path.insert( 0, '../../lib/python3' )

import PUJ

# Load data
data = numpy.loadtxt( open( sys.argv[ 1 ], 'rb' ), delimiter = ',' )

# Configure model
m = PUJ.Model.Linear( )
m.setParameters( [ 0 for i in range( data.shape[ 1 ] ) ] )
print( 'Initial model = ' + str( m ) )

# Configure cost
J = PUJ.Model.Linear.Cost( m, data[ : , 0 : -1 ], data[ : , -1 : ] )

# Fit using an optimization algorithm
opt = PUJ.Optimizer.GradientDescent( J )
opt.setLearningRate( 1e-4 )
opt.Fit( )

## eof - $RCSfile$
