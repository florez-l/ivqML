## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import cv2, mimetypes, numpy, sys

def ReadData( fname ):

  # Guess mime type
  ftype = mimetypes.guess_type( fname, strict = True )[ 0 ]

  # Try different reading strategies, according to mime type
  if ftype == 'image/x-portable-bitmap':

    img = cv2.imread( fname )
    X = numpy.argwhere( img.all( axis = -1, where = False ) )
    Y = img.reshape( ( X.shape[ 0 ], img.shape[ -1 ] ) )

    return [ X, Y ]

  elif ftype == 'text/csv':
    pass
  else:
    pass # ERROR
  # end if
# end def

# We are trying to execute this as a main algorithm
if __name__ == '__main__':

  if len( sys.argv ) < 2:
    print( 'Usage:', sys.argv[ 0 ], '[files]' )
    sys.exit( 1 )
  # end if

  # Read all inputs
  for fname in sys.argv[ 1 : ]:
    ReadData( fname )
  # end for

# end if

## eof - $RCSfile$
