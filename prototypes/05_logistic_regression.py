## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
import Costs, Regressions

# A model
model = Regressions.Logistic( 2 )
model[ 0 ] = 0.1
model[ 1 ] = -2
model[ 2 ] = 30
print( 'Model', model )

# Some values
X = ( numpy.random.rand( 5, model.parameters( ).shape[ 1 ] - 1 ) * 10 ) - 5
print( 'Basic evaluation:\n', model( X ) )
print( 'Basic derivative evaluation:\n', model( X, True ) )
print( 'Threshold evaluation:\n', model.threshold( X ) )

# A cost function
cost = Costs.CrossEntropy( model, X, model.threshold( X ).astype( float ) )
print( 'Cost evaluation:', cost( ) )

## eof - $RCSfile$
