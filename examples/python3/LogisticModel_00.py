## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import os, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )

import numpy
import PUJ.Model.Logistic

m = PUJ.Model.Logistic( [ -1, 2 ] )
x = numpy.random.rand( 30, 1 )

print( m( 0.3 ) )
print( m( 0.3, threshold = False ) )
print( m( x ) )
print( m( x, threshold = False ) )

## eof - $RCSfile$
