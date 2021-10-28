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

# -- Parse command line arguments
parser = PUJ.Helpers.ArgParse( )
parser.add_argument( 'network_descriptor', type = str )
parser.add_argument( 'datafile', type = str )
args = parser.parse_args( )

# -- Data
D = numpy.loadtxt( args.datafile, delimiter = ',' )
numpy.random.shuffle( D )
X, y, *_ = PUJ.Data.Algorithms.SplitData( D, 1 )
X, X_off, X_div = PUJ.Data.Normalize.MinMax( X )

# -- Configure network
nn = PUJ.Model.NeuralNetwork.FeedForward( )
nn.LoadParameters( args.network_descriptor )

# -- Configure cost
cost = PUJ.Model.NeuralNetwork.FeedForward.Cost( X, y, nn, batch_size = 32 )
cost.SetPropagationTypeToBinaryCrossEntropy( )

# -- Prepare debug
debug = PUJ.Debug.Labeling( X, y, threshold = 0.5 )

# -- Iterative solution
PUJ.Optimizer.GradientDescent(
  cost,
  alpha = args.alpha,
  beta1 = args.beta1,
  beta2 = args.beta2,
  max_iter = args.max_iterations,
  epsilon = args.epsilon,
  regularization = args.regularization,
  reg_type = args.reg_type,
  debug_step = args.debug_step,
  debug_function = debug
  )

debug.KeepFigures( )

nn.SaveParameters( 'last_training.nn' )

## eof - $RCSfile$
