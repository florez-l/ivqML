## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================
## This example shows the use of a nD linear regression
## =========================================================================

import numpy, os, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )
import PUJ_ML

# Create model from input data
model = PUJ_ML.Model.NeuralNetwork.FeedForward( input_size = 2 )
model.addLayer( 'ReLU', 8 )
model.addLayer( 'ReLU', 4 )
model.addLayer( 'ReLU', 2 )
model.addLayer( 'Sigmoid', 1 )

print( model )


## eof - $RCSfile$
