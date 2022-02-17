## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import itertools, numpy, sys
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

# Configure cost
J = PUJ.Model.Linear.Cost( m, data[ : , 0 : -1 ], data[ : , -1 : ] )

# Get parameters
ls = numpy.linspace( 0, 3.5, 50 )
for c in itertools.product( ls, repeat = m.numberOfParameters( ) ):
  m.setParameters( list( c ) )
  [ v, g ] = J.evaluate( False )
  print( ' '.join( str( x ) for x in c ) + ' ' + str( v ) )
# end for

## eof - $RCSfile$
