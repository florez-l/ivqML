// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__NeuralNetwork__FeedForward__hxx__
#define __ivqML__Model__NeuralNetwork__FeedForward__hxx__

/* TODO
   #include <functional>
   #include <initializer_list>
   #include <vector>
   #include <ivqML/Model/Base.h>
   #include <ivqML/Model/NeuralNetwork/ActivationFunctions.h>

   #include <algorithm>
   #include <cctype>
   #include <cmath>
   #include <random>
   #include <string>
*/

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TX >
auto ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
operator()( const Eigen::EigenBase< _TX >& X ) const
{
  TNatural M = X.rows( );
  TNatural N = *( std::max_element( this->m_N.begin( ), this->m_N.end( ) ) );
  TReal* buffer
    =
    reinterpret_cast< TReal* >(
      std::calloc( ( N << 1 ) * M, sizeof( TReal ) )
      );
  TMatMap( buffer, M, this->m_N[ 0 ] )
    =
    X.derived( ).template cast< TReal >( );
  this->_eval( buffer, buffer + ( N * M ), M, false );
  TMat A = TMatMap( buffer, M, this->output_size( ) );
  std::free( buffer );
  return( A );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TX, class _Ty >
void  ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
fit(
  const Eigen::EigenBase< _TX >& bX,
  const Eigen::EigenBase< _Ty >& by,
  const TReal& L1, const TReal& L2
  )
{
  /* TODO
     if( n == 0 || m != y.rows( ) )
     throw AssertionError( 'There is no closed solution for a logistic regression.' )
  */
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TG, class _TX, class _Ty >
typename ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
TReal ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
cost_gradient(
  Eigen::EigenBase< _TG >& G,
  const Eigen::EigenBase< _TX >& bX,
  const Eigen::EigenBase< _Ty >& by,
  const TReal& L1, const TReal& L2
  )
{
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TX, class _Ty >
typename ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
TReal ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
cost( const Eigen::EigenBase< _TX >& X, const Eigen::EigenBase< _Ty >& y )
{
}

#endif // __ivqML__Model__NeuralNetwork__FeedForward__hxx__

// eof - $RCSfile$
