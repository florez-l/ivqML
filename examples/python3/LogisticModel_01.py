## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import sys
sys.path.insert( 0, '../../lib/python3' )

import PUJ.Model.Logistic

m = PUJ.Model.Logistic( )
m.setParameters( [ 0, 1, 0 ] )
print( m )

print( m.evaluate( [ 2, 4 ] ) )
print( m.threshold( [ 2, 4 ] ) )

## eof - $RCSfile$
