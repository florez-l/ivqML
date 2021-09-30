## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import os, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )

import numpy
import PUJ.Model.Linear

m = PUJ.Model.Linear( bias = 1, weights = 2 )
n = PUJ.Model.Linear( parameters = [ 1, 2 ] )

x = numpy.random.uniform( -10, 10, ( 10, 1 ) )

print( m( 3.141592 ) )
print( n( 3.141592 ) )
print( m( x ) )

## eof - $RCSfile$
