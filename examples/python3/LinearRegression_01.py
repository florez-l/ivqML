## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import argparse, numpy, os, random, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )

import PUJ.Helpers.ArgParse
import PUJ.Data.Algorithms
import PUJ.Model.Linear
import PUJ.Regression.MSE
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

## -- Prepare regression
cost = PUJ.Regression.MSE( X, y )

## -- Analitical solution
tA = cost.AnalyticSolve( )

## -- Init parameters
if args.init == 'zeros':
  t0 = numpy.zeros( X.shape[ 1 ] )
elif args.init == 'ones':
  t0 = numpy.ones( X.shape[ 1 ] )
else:
  t0 = numpy.random.uniform( low = 0, high = 1, size = X.shape[ 1 ] )
# end if

## -- Prepare debug
debug = PUJ.Debug.Polynomial( X, y )

## -- Iterative solution
tI, nI = PUJ.Optimizer.GradientDescent(
  cost,
  learning_rate = args.learning_rate,
  init_theta = t0,
  max_iter = args.max_iterations,
  epsilon = args.epsilon,
  debug_step = args.debug_step,
  debug_function = debug
  )

## -- Show results
tD = tI - tA
print( '=================================================================' )
print( '* Analitical solution  :', tA )
print( '* Iterative solution   :', tI )
print( '* Difference           :', ( tD @ tD.T )[ 0 , 0 ] ** 0.5 )
print( '* Final cost           :', cost.Cost( tI ) )
print( '* Number of iterations :', nI )
print( '=================================================================' )

debug.KeepFigures( )

## eof - $RCSfile$
