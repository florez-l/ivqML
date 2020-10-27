## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import itertools, math, numpy

## -------------------------------------------------------------------------
def Label( X, means ):
  n = X.shape[ 0 ]
  k = means.shape[ 0 ]
  D = numpy.argmin(
        numpy.array(
          [
            numpy.linalg.norm( X[ r, : ] - means, axis = 1 )
            for r in range( n )
          ]
        ), axis = 1
      )
  return D
# end def

## -------------------------------------------------------------------------
def BruteForce( X, k ):
  n = X.shape[ 0 ]
  J = math.inf
  means = None
  iteration = 0
  for c in itertools.combinations( range( n ), k ):
    M = numpy.matrix( [ X[ c[ i ], : ] for i in range( k ) ] )
    D = numpy.argmin(
          numpy.array(
            [
              numpy.linalg.norm( X[ r, : ] - M, axis = 1 )
              for r in range( n )
            ]
          ), axis = 1
        )
    Mu = numpy.matrix(
           [
             numpy.mean( X[ ( D == i ), : ], axis = 0 )
             for i in range( k )
           ]
         )
    q = numpy.array(
          [
            numpy.linalg.norm(
              X[ ( D == i ), : ] - Mu[ i, : ], axis = 1, ord = 2
            ).sum( )
            for i in range( k )
          ]
        ).sum( )
    if q < J:
      J = q
      means = Mu
    # end if
    if iteration % 1000 == 0:
      print( "Iteration:", c, q, J )
    # end if
    iteration += 1
  # end for
  return [ Label( X, means ), means ]
# end def

## -------------------------------------------------------------------------
def Lloyd( X, init_means ):
  n = X.shape[ 0 ]
  m = X.shape[ 1 ]
  k = init_means.shape[ 0 ]
  mu = numpy.copy( init_means )
  iteration = 0
  stop = False
  while not stop:
    D = numpy.argmin(
          numpy.array(
            [
              numpy.linalg.norm( X[ r, : ] - mu, axis = 1 )
              for r in range( n )
            ]
          ), axis = 1
        )
    mu2 = numpy.matrix(
           [
             numpy.mean( X[ ( D == i ), : ], axis = 0 )
             for i in range( k )
           ]
         )
    stop = ( numpy.linalg.norm( mu - mu2 ) < 1e-7 )
    mu = mu2
    iteration += 1
  # end while
  return [ Label( X, mu ), mu ]
# end def

## -------------------------------------------------------------------------
# Adapted from:
# ---> https://gist.github.com/kweimann/5d4a0a0a33acb39fe4c3334e682790cd
def GaussianEM( X, k, eps = 1e-7 ):
  m, n = X.shape
  last_log_estimate = None
  mean = numpy.random.uniform( numpy.min( X, axis = 0 ), numpy.max( X, axis = 0 ), size = ( k, n ) )
  cov = [ numpy.identity( n ) for i in range( k ) ]
  p = numpy.full( ( k, ), 1.0 / k )

  stop = False
  while not stop:
    # Expectation
    p_cluster = numpy.zeros( ( m, k ) )
    for c in range( k ):
      var = numpy.sum( ( X - mean[ c ] ).dot( numpy.linalg.inv( cov[ c ] ) ) * ( X - mean[ c ] ), axis = 1 )
      p_cluster[ :, c ] = p[ c ] * numpy.exp( -var / 2 ) / numpy.sqrt( numpy.abs( 2 * numpy.pi * numpy.linalg.det( cov[ c ] ) ) )
      normalized_p_cluster = p_cluster / numpy.sum( p_cluster, axis = 1 ).reshape( ( -1, 1 ) )
    # end for

    # Maximization
    p = numpy.sum( normalized_p_cluster, axis = 0 ) / m
    for c in range( k ):
      mean[ c ] = numpy.average( X, axis = 0, weights=normalized_p_cluster[ :, c ] )
      cov[ c ] = numpy.cov( X.T, aweights = normalized_p_cluster[ :, c ] )
    # end for

    # Log estimate
    log_estimate = numpy.sum( numpy.log( numpy.sum( p_cluster, axis = 0 ) ) )
    if last_log_estimate is not None and numpy.abs( log_estimate - last_log_estimate ) < eps:
      stop = True
    else:
      last_log_estimate = log_estimate
    # end if
  # end while
  return numpy.argmax( normalized_p_cluster, axis = 1 )
# end def

## eof - $RCSfile$
