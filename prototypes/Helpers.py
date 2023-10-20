## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, random

# --------------------------------------------------------------------------
def extend_polynomial( X, n ):
  for i in range( n - 1 ):
    X = \
      numpy.concatenate( \
        ( X, \
          numpy.reshape( \
              numpy.multiply( X[ : , 0 ], X[ : , i ] ), \
              ( X.shape[ 0 ], 1 ) ) \
          ), axis = 1 \
        )
  # end for
  return X
# end def

# --------------------------------------------------------------------------
def extract_data_from_grayscaleimage( I, n_samples, classes = 2 ):
  n = I.shape[ 0 ]
  for i in range( 1, len( I.shape ) - 1 ):
    n *= I.shape[ i ]
  # end for
  JY = I.astype( float ).mean( axis = -1 ).reshape( ( n, 1 ) )

  mJ = JY.min( )
  MJ = JY.max( )
  JY = ( ( JY - mJ ) / ( ( MJ - mJ ) / float( classes - 1 ) ) ).astype( int )

  idxX, idxY = \
    numpy.meshgrid( \
      numpy.linspace( 0, I.shape[ 1 ] - 1, I.shape[ 1 ] ), \
      numpy.linspace( 0, I.shape[ 0 ] - 1, I.shape[ 0 ] ) \
      )
  JX = \
    numpy.concatenate( \
      ( \
        numpy.reshape( idxY, ( idxY.shape[ 0 ] * idxY.shape[ 1 ], 1 ) ), \
        numpy.reshape( idxX, ( idxX.shape[ 0 ] * idxX.shape[ 1 ], 1 ) ) \
      ), \
      axis = 1 \
      )

  idx = []
  for c in range( classes ):
    i = ( JY == c ).nonzero( )[ 0 ].tolist( )
    random.shuffle( i )
    idx += i[ : n_samples ]
  # end for

  D = numpy.concatenate( ( JX[ idx , : ], JY[ idx , : ] ), axis = 1 )
  idx = list( range( D.shape[ 0 ] ) )
  ## random.shuffle( idx )

  return ( D[ idx , : -1 ],  D[ idx , -1 : ] )
# end def

## eof - $RCSfile$
