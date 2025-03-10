// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <ivqML/Model/NeuralNetwork/ActivationFunctions.h>

// -------------------------------------------------------------------------
template< class _TReal >
typename ivqML::Model::NeuralNetwork::ActivationFunctions< _TReal >::
TFunction ivqML::Model::NeuralNetwork::ActivationFunctions< _TReal >::
Get( const std::string& a )
{
  std::string n = a;
  std::transform(
    n.begin( ), n.end( ), n.begin( ),
    []( const unsigned char& c ) -> unsigned char
    {
      return( std::tolower( c ) );
    }
    );
  if( n == "softmax" )
  {
    // TODO
    return( []( TMap& A, const TMap& Z, bool d ) -> void {} );
  }
  else
  {
    if( n == "relu" )
      return(
        []( TMap& A, const TMap& Z, bool d ) -> void
        {
          A = Z.unaryExpr(
            [&d]( const TReal& z ) -> TReal
            {
              if( d )
                return( ( z < TReal( 0 ) )? TReal( 0 ): TReal( 1 ) );
              else
                return( ( z < TReal( 0 ) )? TReal( 0 ): z );
            }
            );
        }
        );
    else if( n == "tanh" )
      return(
        []( TMap& A, const TMap& Z, bool d ) -> void
        {
          A = Z.unaryExpr(
            [&d]( const TReal& z ) -> TReal
            {
              TReal a = std::tanh( z );
              if( d )
                return( TReal( 1 ) - ( a * a ) );
              else
                return( a );
            }
            );
        }
        );
    else if( n == "sigmoid" )
      return(
        []( TMap& A, const TMap& Z, bool d ) -> void
        {
          A = Z.unaryExpr(
            [&d]( const TReal& z ) -> TReal
            {
              static const TReal _0  = TReal( 0 );
              static const TReal _1  = TReal( 1 );
              static const TReal _M  = std::numeric_limits< TReal >::max( );
              static const TReal _L  = std::log( _M ) / TReal( 2 );

              TReal s;
              if     ( z >  _L ) s = _1;
              else if( z < -_L ) s = _0;
              else               s = _1 / ( _1 + std::exp( -z ) );

              return( s * ( ( d )? ( _1 - s ): _1 ) );
            }
            );
        }
        );
    else // if( n == "identity" )
      return(
        []( TMap& A, const TMap& Z, bool d ) -> void
        {
          A = Z.unaryExpr(
            [&d]( const TReal& z ) -> TReal
            {
              return( ( d )? TReal( 1 ): z );
            }
            );
        }
        );
  } // end if
}

// -------------------------------------------------------------------------
namespace ivqML
{
  namespace Model
  {
    namespace NeuralNetwork
    {
      template class ivqML_EXPORT ActivationFunctions< float >;
      template class ivqML_EXPORT ActivationFunctions< double >;
      template class ivqML_EXPORT ActivationFunctions< long double >;
    } // end namespace
  } // end namespace
} // end namespace

// eof - $RCSfile$
