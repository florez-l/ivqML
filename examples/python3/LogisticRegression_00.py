## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import argparse, numpy, os, random, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )

import PUJ.Helpers.ArgParse
import PUJ.Data.Normalize
import PUJ.Optimizer.GradientDescent
import PUJ.Debug.Labeling

## -- Parse command line arguments
parser = PUJ.Helpers.ArgParse( )
parser.add_argument( 'filename', type = str )
args = parser.parse_args( )

# -- Data
D = numpy.loadtxt( args.filename, delimiter = ',' )
numpy.random.shuffle( D )
X, y, *_ = PUJ.Data.Algorithms.SplitData( D, 1 )
X, X_off, X_div = PUJ.Data.Normalize.Center( X )

## -- Prepare regression
model = PUJ.Model.Logistic(
  parameters = args.init,
  size = X.shape[ 1 ]
  )
cost = PUJ.Model.Logistic.Cost( X, y, model )

## -- Prepare debug
debug = PUJ.Debug.Labeling( X, y )

## -- Iterative solution
PUJ.Optimizer.GradientDescent(
  cost,
  learning_rate = args.learning_rate,
  max_iter = args.max_iterations,
  epsilon = args.epsilon,
  debug_step = args.debug_step,
  debug_function = debug
  )

## -- Show results
K, acc = PUJ.Data.Algorithms.Accuracy( y, model( X, threshold = True ) )
print( '=================================================================' )
print( '* Solution             :', model )
print( '* Number of iterations :', debug.GetNumberOfIterations( ) )
print( '* Accuracy             :', acc )
print( '* Confusion matrix     :' )
print( K )
print( '=================================================================' )
debug.KeepFigures( )

## eof - $RCSfile$
