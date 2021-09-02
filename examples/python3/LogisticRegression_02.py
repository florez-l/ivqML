## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import argparse, numpy, os, random, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )

import PUJ.Regression.MaximumLikelihood
import PUJ.Data.Normalize, PUJ.Optimizer.GradientDescent
import PUJ.Debug.Cost

## -- Parse command line arguments
parser = argparse.ArgumentParser( )
parser.add_argument( 'filename', type = str )
parser.add_argument( '-a', '--learning_rate', type = float )
parser.add_argument( '-I', '--max_iterations', type = int )
parser.add_argument( '-D', '--debug_step', type = int )
parser.add_argument( '-e', '--epsilon', type = float )
parser.add_argument( '--init', type = str, choices = [ 'zeros', 'ones', 'random' ] )
args = parser.parse_args( )

lr = 1e-2
I = 1000
e = 1e-8
d = 10
init = 'zeros'

if args.learning_rate != None: lr = args.learning_rate
if args.max_iterations != None: I = args.max_iterations
if args.debug_step != None: d = args.debug_step
if args.epsilon != None: e = args.epsilon
if args.init != None: init = args.init

# -- Data
reader = PUJ.Data.Reader( train_size = 0.6, test_size = 0.3 )
D = reader.FromCSV( args.filename, output_size = 1, shuffle = True )
Xtra, X_off, X_div = PUJ.Data.Normalize.Standardize( D[ 0 ] )
Xtst = ( D[ 2 ] - X_off ) / X_div
Xval = ( D[ 4 ] - X_off ) / X_div

ytra = D[ 1 ]
ytst = D[ 3 ]
yval = D[ 5 ]

# -- Initial parameters
if init == 'zeros':
  init_theta = numpy.zeros( Xtra.shape[ 1 ] + 1 )
elif init == 'ones':
  init_theta = numpy.ones( Xtra.shape[ 1 ] + 1 )
else:
  init_theta = \
    numpy.random.uniform( low = -1.0, high = 1.0, size = Xtra.shape[ 1 ] + 1 )
# end if

# -- Prepare regression
cost_tra = PUJ.Regression.MaximumLikelihood( Xtra, ytra )
cost_tst = PUJ.Regression.MaximumLikelihood( Xtst, ytst )
cost_val = PUJ.Regression.MaximumLikelihood( Xval, yval )

# -- Prepare debug
debug = PUJ.Debug.Cost( cost_tst )

# -- Iterative solution
tI, nI = PUJ.Optimizer.GradientDescent(
  cost_tra,
  learning_rate = lr,
  init_theta = init_theta,
  max_iter = I,
  epsilon = e,
  debug_step = d,
  debug_function = debug
  )

print( '=================================================================' )
print( '* Normalization offset  :', X_off )
print( '* Normalization scale   :', X_div )
print( '* Solution              :', tI )
print( '* Number of iterations  :', nI )
print( '* Final training cost   :', cost_tra.Cost( tI ) )
print( '* Final testing cost    :', cost_tst.Cost( tI ) )
print( '* Final validation cost :', cost_val.Cost( tI ) )
print( '=================================================================' )

debug.KeepFigures( )

## eof - $RCSfile$
