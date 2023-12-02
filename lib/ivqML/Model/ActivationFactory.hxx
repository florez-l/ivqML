// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__ActivationFactory__hxx__
#define __ivqML__Model__ActivationFactory__hxx__

#include <algorithm>
#include <cctype>
#include <cmath>

// -------------------------------------------------------------------------
template< class _M >
typename ivqML::Model::ActivationFactory< _M >::
TActivation ivqML::Model::ActivationFactory< _M >::
New( const std::string& n )
{
  std::string rn = n;
  std::transform(
    rn.begin( ), rn.end( ), rn.begin( ),
    []( unsigned char c ){ return( std::tolower( c ) ); }
    );

  if( rn == "relu" )
    return(
      []( TMap& A, const TMap& Z, bool d ) -> void
      {
        A = Z.unaryExpr(
          [&d]( const TScalar& z ) -> TScalar
          {
            return( ( z < s_0 )? s_0: ( d? s_1: z ) );
          }
          );
      }
      );
  else if( rn == "leakyrelu" )
    return(
      []( TMap& A, const TMap& Z, bool d ) -> void
      {
        A = Z.unaryExpr(
          [&d]( const TScalar& z ) -> TScalar
          {
            return(
              ( z < s_0 )
              ?
              ( d? TScalar( 1e-2 ): z * TScalar( 1e-2 ) )
              :
              ( d? s_1: z )
              );
          }
          );
      }
      );
  else if( rn == "tanh" )
    return(
      []( TMap& A, const TMap& Z, bool d ) -> void
      {
        A = Z.unaryExpr(
          [&d]( const TScalar& z ) -> TScalar
          {
            TScalar t = std::tanh( z );
            return( d? ( s_1 - ( t * t ) ): t );
          }
          );
      }
      );
  else if( rn == "sigmoid" )
    return(
      []( TMap& A, const TMap& Z, bool d ) -> void
      {
        A = Z.unaryExpr(
          [&d]( const TScalar& z ) -> TScalar
          {
            TScalar a;
            if( z <= -s_L )
              a = s_0;
            else if( z >= s_L )
              a = s_1;
            else
              a = s_1 / ( s_1 + std::exp( -z ) );
            return( d? ( a * ( s_1 - a ) ): a );
          }
          );
      }
      );
  else if( rn == "softmax" )
    return(
      []( TMap& A, const TMap& Z, bool d ) -> void
      {
        A = ( Z.colwise( ) - Z.rowwise( ).maxCoeff( ) ).array( ).exp( );
        A.array( ).colwise( ) /= A.array( ).rowwise( ).sum( );
      }
      );
  else // if( "linear" )
    return(
      []( TMap& A, const TMap& Z, bool d ) -> void
      {
        A = Z.unaryExpr(
          [&d]( const TScalar& z ) -> TScalar
          {
            return( d? s_1: z );
          }
          );
      }
      );
}

#endif // __ivqML__Model__ActivationFactory__hxx__

// eof - $RCSfile$
