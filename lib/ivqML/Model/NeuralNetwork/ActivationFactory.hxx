// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__NeuralNetwork__ActivationFactory__hxx__
#define __ivqML__Model__NeuralNetwork__ActivationFactory__hxx__

#include <algorithm>
#include <cctype>
#include <cmath>

// -------------------------------------------------------------------------
template< class _M >
typename ivqML::Model::NeuralNetwork::ActivationFactory< _M >::
TActivation ivqML::Model::NeuralNetwork::ActivationFactory< _M >::
New( const std::string& n )
{
  static const TScl _0 { TScl( 0 ) };
  static const TScl _1 { TScl( 1 ) };
  static const TScl _E { TTraits::epsilon( ) };
  static const TScl _L = std::log( _1 - _E ) - std::log( _E );

  std::string rn = n;
  std::transform(
    rn.begin( ), rn.end( ), rn.begin( ),
    []( unsigned char c ){ return( std::tolower( c ) ); }
    );

  if( rn == "relu" )
    return(
      []( TMatMap& A, const TMatMap& Z, bool d ) -> void
      {
        A = Z.unaryExpr(
          [&d]( const TScl& z ) -> TScl
          {
            return( ( z < _0 )? _0: ( d? _1: z ) );
          }
          );
      }
      );
  else if( rn == "leakyrelu" )
    return(
      []( TMatMap& A, const TMatMap& Z, bool d ) -> void
      {
        A = Z.unaryExpr(
          [&d]( const TScl& z ) -> TScl
          {
            return(
              ( z < _0 )
              ?
              ( d? TScl( 1e-2 ): z * TScl( 1e-2 ) )
              :
              ( d? _1: z )
              );
          }
          );
      }
      );
  else if( rn == "tanh" )
    return(
      []( TMatMap& A, const TMatMap& Z, bool d ) -> void
      {
        A = Z.unaryExpr(
          [&d]( const TScl& z ) -> TScl
          {
            TScl t = std::tanh( z );
            return( d? ( _1 - ( t * t ) ): t );
          }
          );
      }
      );
  else if( rn == "sigmoid" )
    return(
      []( TMatMap& A, const TMatMap& Z, bool d ) -> void
      {
        A = Z.unaryExpr(
          [&]( const TScl& z ) -> TScl
          {
            TScl a;
            if     ( z <= -_L ) a = _0;
            else if( z >=  _L ) a = _1;
            else                a = _1 / ( _1 + std::exp( -z ) );
            return( d? ( a * ( _1 - a ) ): a );
          }
          );
      }
      );
  else if( rn == "softmax" )
    return(
      []( TMatMap& A, const TMatMap& Z, bool d ) -> void
      {
        A = ( Z.rowwise( ) - Z.colwise( ).maxCoeff( ) ).array( ).exp( ).eval( );
        A.array( ).rowwise( ) /= A.array( ).colwise( ).sum( );
      }
      );
  else // if( "linear" )
    return(
      []( TMatMap& A, const TMatMap& Z, bool d ) -> void
      {
        A = Z.unaryExpr(
          [&d]( const TScl& z ) -> TScl
          {
            return( d? _1: z );
          }
          );
      }
      );
}

#endif // __ivqML__Model__NeuralNetwork__ActivationFactory__hxx__

// eof - $RCSfile$
