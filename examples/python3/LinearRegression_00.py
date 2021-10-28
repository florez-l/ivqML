## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import argparse, numpy, os, random, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )

import PUJ.Helpers.ArgParse
import PUJ.Data.Algorithms
import PUJ.Model.Linear
import PUJ.Optimizer.GradientDescent
import PUJ.Debug.Polynomial

## -- Parse command line arguments
parser = PUJ.Helpers.ArgParse( )
parser.add_argument(
  'coefficients', metavar = 'W', type = str, nargs = '+',
  help = 'list of polynomial coefficients (wx float)'
  )
parser.add_argument( '-n', '--number_of_samples', type = int, default = 100 )
parser.add_argument( '-v0', type = float, default = -10 )
parser.add_argument( '-v1', type = float, default = 10 )
args = parser.parse_args( )

## -- Create polynomial
n = max(
  [
    int( args.coefficients[ i ][ 1 : ] )
    for i in range( 0, len( args.coefficients ), 2 )
    ]
  )
w_true = [ 0.0 for i in range( n + 1 ) ]
for i in range( 0, len( args.coefficients ), 2 ):
  w_true[ int( args.coefficients[ i ][ 1 : ] ) ] = \
          float( args.coefficients[ i + 1 ] )
# end for
model = PUJ.Model.Linear( parameters = w_true )

## -- Synthetic data
vOff = ( args.v1 - args.v0 ) / args.number_of_samples
X = numpy.matrix( numpy.arange( args.v0, args.v1 + 1, vOff ) ).T
for i in range( 1, model.GetInputSize( ) ):
  X = numpy.append(
    X,
    numpy.array( X[ : , 0 ] ) * numpy.array( X[ : , i - 1 ] ),
    axis = 1
    )
# end for
D = numpy.append( X, model( X ), axis = 1 )
numpy.random.shuffle( D )
X, y, *_ = PUJ.Data.Algorithms.SplitData( D, 1 )

## -- Analitical solution
analitical_model = PUJ.Model.Linear( )
analitical_model.Fit( X, y )

## -- Prepare regression
iterative_model = PUJ.Model.Linear(
  parameters = args.init,
  size = model.GetInputSize( )
  )
cost = PUJ.Model.Linear.Cost( X, y, iterative_model )

## -- Prepare debug
debug = PUJ.Debug.Polynomial( X, y )

## -- Iterative solution
PUJ.Optimizer.GradientDescent(
  cost,
  alpha = args.alpha,
  max_iter = args.max_iterations,
  epsilon = args.epsilon,
  debug_step = args.debug_step,
  debug_function = debug
  )

## -- Show results
print( '=================================================================' )
print( '* Analitical solution :', analitical_model )
print( '* Iterative solution  :', iterative_model )
print( '* Number of iterations :', debug.GetNumberOfIterations( ) )
print( '=================================================================' )
debug.KeepFigures( )

## eof - $RCSfile$
