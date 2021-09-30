## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, os, random, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )

import PUJ.Helpers.ArgParse
import PUJ.Data.Normalize
import PUJ.Model.NeuralNetwork.FeedForward
import PUJ.Optimizer.GradientDescent
import PUJ.Debug.Labeling

## -- Parse command line arguments
parser = PUJ.Helpers.ArgParse( )
parser.add_argument( 'network_descriptor', type = str )
parser.add_argument( 'datafile', type = str )
args = parser.parse_args( )

# -- Data
D = numpy.loadtxt( args.datafile, delimiter = ',' )
numpy.random.shuffle( D )
X, y, *_ = PUJ.Data.Algorithms.SplitData( D, 1 )
X, X_off, X_div = PUJ.Data.Normalize.Center( X )

# -- Configure network
nn = PUJ.Model.NeuralNetwork.FeedForward( )
nn.LoadParameters( args.network_descriptor )

# -- Configure cost
cost = PUJ.Model.NeuralNetwork.FeedForward.Cost( X, y, nn )
cost.SetPropagationTypeToBinaryCrossEntropy( )

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
#K, acc = PUJ.Data.Algorithms.Accuracy( y, nn( X, threshold = True ) )
#print( '=================================================================' )
#print( '* Solution             :', model )
#print( '* Number of iterations :', debug.GetNumberOfIterations( ) )
#print( '* Accuracy             : {:.1f}%'.format( acc * 100 ) )
#print( '* Confusion matrix     :' )
#print( K )
#print( '=================================================================' )
debug.KeepFigures( )

##import numpy
##import PUJ.Data.Algorithms, PUJ.Data.Normalize

##
##
##def debug( J, dJ, t, i, show ):
##  if show:
##    print( i, dJ )
##  # end if
### end def
##
### -- Configure network
##
### -- Data
##D = numpy.loadtxt( args.filename, delimiter = ',' )
##numpy.random.shuffle( D )
##X, y, *_ = PUJ.Data.Algorithms.SplitData( D, 1 )
##X, X_off, X_div = PUJ.Data.Normalize.Center( X )
##
### -- Train
##cost_tra = PUJ.Model.NeuralNetwork.FeedForward.Cost( X_real, y_real, fwd_nn )
##cost_tra.SetPropagationTypeToBinaryCrossEntropy( )
##
### -- Iterative solution
##tI, nI = PUJ.Optimizer.GradientDescent(
##  cost_tra,
##  learning_rate = 1e-4,
##  init_theta = 'none',
##  max_iter = 100000,
##  epsilon = 1e-8,
##  debug_step = 1000,
##  debug_function = debug
##  )
##
##print( '===================================' )
##print( fwd_nn )
##print( '===================================' )

## fwd_nn.SaveParameters( 'leo.nn' )

## y_estim = ( fwd_nn( X_real ) >= 0.5 ).astype( D.dtype )
## K, acc = PUJ.Data.Algorithms.Accuracy( y_real, y_estim )

## print( K )
## print( 'Accuracy: ' + str( acc * 100 ) + '%' )


## eof - $RCSfile$
