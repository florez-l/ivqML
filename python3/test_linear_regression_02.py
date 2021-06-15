## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
import Cost, GradientDescent

m = 10000
s = 0
W_perfect = numpy.matrix( [ 2, 1 ] )
b_perfect = float( 1.5 )

X1_input = numpy.matrix( numpy.random.uniform( -100, 100, ( m, 1 ) ) )
X2_input = numpy.array( X1_input ) * numpy.array( X1_input )
X_input = numpy.append( X1_input, X2_input, axis = 1 )
y_input = numpy.matrix(
    numpy.random.normal( ( X_input @ W_perfect.T ) + b_perfect, s )
    )

cost = Cost.MSE( X_input, y_input )
[ W_gd, b_gd, nIter ] = GradientDescent.Solve(
    cost,
    learning_rate = 1e-1,
    max_iterations = 1e8,
    epsilon = 1e-16,
    debug_step = 10000
    )
[ W_as, b_as ] = cost.AnalyticSolve( )

print( '**********************************************' )
print( 'Input parameters:', W_perfect, b_perfect )
print( 'Gradient descent:', W_gd, b_gd, '(' + str( nIter ) + ')' )
print( 'Analytical      :', W_as, b_as )
print( '**********************************************' )

data = ''
for i in range( X_input.shape[ 0 ] ):
  data += str( X_input[ i, 0 ] ) + ' ' + str( y_input[ i, 0 ] ) + '\n'
# end for
out_data = open( 'linear_regression_data.txt', 'w' )
out_data.write( data )
out_data.close( )

## eof - $RCSfile$
