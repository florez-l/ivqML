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
parser.add_argument( 'filename', type = str )
args = parser.parse_args( )

## -- Data
D = numpy.loadtxt( args.filename, delimiter = ',' )
numpy.random.shuffle( D )
X, y, *_ = PUJ.Data.Algorithms.SplitData( D, 1 )

## -- Analitical solution
analitical_model = PUJ.Model.Linear( )
analitical_model.Fit( X, y )

## -- Prepare regression
iterative_model = PUJ.Model.Linear(
  parameters = args.init,
  size = X.shape[ 1 ]
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
