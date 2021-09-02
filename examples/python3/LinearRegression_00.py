## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import argparse, numpy, os, random, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )

import PUJ.Model.Linear, PUJ.Regression.MSE
import PUJ.Data.Normalize, PUJ.Optimizer.GradientDescent
import PUJ.Debug.Polynomial

## -- Parse command line arguments
parser = argparse.ArgumentParser( )
parser.add_argument(
  'coefficients', metavar = 'W', type = str, nargs = '+',
  help = 'list of polynomial coefficients (wx float)'
  )
parser.add_argument( '-n', '--number_of_samples', type = int )
parser.add_argument( '-v0', type = float )
parser.add_argument( '-v1', type = float )
parser.add_argument( '-a', '--learning_rate', type = float )
parser.add_argument( '-I', '--max_iterations', type = int )
parser.add_argument( '-D', '--debug_step', type = int )
parser.add_argument( '-e', '--epsilon', type = float )
parser.add_argument( '--init', type = str, choices = [ 'zeros', 'ones', 'random' ] )
args = parser.parse_args( )

m = 20
v0 = -10
v1 =  10
lr = 1e-2
I = 1000
e = 1e-8
d = 10
init = 'zeros'

if args.number_of_samples != None: m = args.number_of_samples
if args.v0 != None: v0 = args.v0
if args.v1 != None: v1 = args.v1
if args.learning_rate != None: lr = args.learning_rate
if args.max_iterations != None: I = args.max_iterations
if args.debug_step != None: d = args.debug_step
if args.epsilon != None: e = args.epsilon
if args.init != None: init = args.init

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
## end for

if init == 'zeros':
  init_theta = numpy.zeros( len( w_true ) )
elif init == 'ones':
  init_theta = numpy.ones( len( w_true ) )
else:
  init_theta = \
    numpy.random.uniform( low = -1.0, high = 1.0, size = len( w_true ) )
# end if

# -- Synthetic model
model = PUJ.Model.Linear( w_true )

# -- Synthetic data
vOff = ( v1 - v0 ) / m
x0 = numpy.matrix( numpy.arange( v0, v1 + 1, vOff ) ).T
for i in range( 1, model.Dimensions( ) ):
  x0 = numpy.append(
    x0,
    numpy.array( x0[ : , 0 ] ) * numpy.array( x0[ : , i - 1 ] ),
    axis = 1
    )
# end for
D = numpy.append( x0, model( x0 ), axis = 1 )
numpy.random.shuffle( D )
x0 = D[ : , : -1 ]
y0 = D[ : , -1 : ]

# -- Prepare regression
cost = PUJ.Regression.MSE( x0, y0 )

# -- Analitical solution
tA = cost.AnalyticSolve( )

# -- Prepare debug
debug = PUJ.Debug.Polynomial( x0[ : , 0 ], y0 )

# -- Iterative solution
tI, nI = PUJ.Optimizer.GradientDescent(
  cost,
  learning_rate = lr,
  init_theta = init_theta,
  max_iter = I,
  epsilon = e,
  debug_step = d,
  debug_function = debug
  )

print( '=================================================================' )
print( '* Analitical solution  :', tA )
print( '* Iterative solution   :', tI )
print(
  '* Difference           :',
  ( ( tI - tA ) @ ( tI - tA ).T )[ 0, 0 ] ** 0.5
  )
print( '* Final cost           :', cost.Cost( tI ) )
print( '* Number of iterations :', nI )
print( '=================================================================' )

debug.KeepFigures( )

## eof - $RCSfile$
