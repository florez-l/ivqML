// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Model/Functions.h>
#include <cctype>
#include <cmath>

// -------------------------------------------------------------------------
template< class _TReal >
typename ivqML::Model::Functions< _TReal >::
TPair ivqML::Model::Functions< _TReal >::
Get( const std::string& name )
{
  auto _lwr = []( const std::string& s ) -> std::string
    {
      std::string r = s;
      std::transform(
        r.begin( ), r.end( ), r.begin( ),
        []( const unsigned char& c )
        {
          return( std::tolower( c ) );
        }
        );
      return( r );
    };
  std::string lname = _lwr( name );

  if( lname == "softmax" )
  {
    return(
      std::make_pair(
        "softmax",
        []( TMap& A, const TMap& Z, bool d ) -> void
        {
          TMatrix m = Z.colwise( ).maxCoeff( );
          A = ( Z.rowwise( ) - m.row( 0 ) ).array( ).exp( );
          m = A.colwise( ).sum( );
          A.array( ).rowwise( ) /= m.array( ).row( 0 );
          if( d )
            A
              =
              A.unaryExpr(
                []( const TReal& a ) -> TReal
                {
                  return( a * ( TReal( 1 ) - a ) );
                }
                );
        }
        )
      );
  }
  else
  {
    if( lname == "relu" )
    {
      return(
        std::make_pair(
          "relu",
          []( TMap& A, const TMap& Z, bool d ) -> void
          {
            A = Z.unaryExpr(
              [&d]( const TReal& z ) -> TReal
              {
                if( d )
                  return( TReal( ( z < TReal( 0 ) )? 0: 1 ) );
                else
                  return( ( z < TReal( 0 ) )? TReal( 0 ): z );
              }
              );
          }
          )
        );
    }
    else if( lname == "leakyrelu" )
    {
      return(
        std::make_pair(
          "leakyrelu",
          []( TMap& A, const TMap& Z, bool d ) -> void
          {
            A = Z.unaryExpr(
              [&d]( const TReal& z ) -> TReal
              {
                if( d )
                  return( ( z < TReal( 0 ) )? TReal( 1e-2 ): TReal( 1 ) );
                else
                  return(
                    z * ( ( z < TReal( 0 ) )? TReal( 1e-2 ): TReal( 1 ) )
                    );
              }
              );
          }
          )
        );
    }
    else if( lname == "tanh" )
    {
      return(
        std::make_pair(
          "tanh",
          []( TMap& A, const TMap& Z, bool d ) -> void
          {
            A = Z.unaryExpr(
              [&d]( const TReal& z ) -> TReal
              {
                TReal a = std::tanh( z );
                return( ( d )? ( TReal( 1 ) - ( a * a ) ): a );
              }
              );
          }
          )
        );
    }
    else if( lname == "sigmoid" )
    {
      static const TReal M  = std::numeric_limits< TReal >::max( );
      static const TReal L  = std::log( M ) / TReal( 2 );
      return(
        std::make_pair(
          "sigmoid",
          [&]( TMap& A, const TMap& Z, bool d ) -> void
          {
            A = Z.unaryExpr(
              [&d]( const TReal& z ) -> TReal
              {
                TReal a;
                if     ( z < -L ) a = TReal( 0 );
                else if( L < z  ) a = TReal( 1 );
                else a = TReal( 1 ) / ( TReal( 1 ) + std::exp( -z ) );
                return( ( d )? ( a * ( TReal( 1 ) - a ) ): a );
              }
              );
          }
          )
        );
    }
    else // if( lname == "linear" )
    {
      return(
        std::make_pair(
          "linear",
          []( TMap& A, const TMap& Z, bool d ) -> void
          {
            A = Z.unaryExpr(
              [&d]( const TReal& z ) -> TReal
              {
                return( ( d )? TReal( 1 ): z );
              }
              );
          }
          )
        );
    } // end if
  } // end if
}

// -------------------------------------------------------------------------
template class ivqML_EXPORT ivqML::Model::Functions< float >;
template class ivqML_EXPORT ivqML::Model::Functions< double >;
template class ivqML_EXPORT ivqML::Model::Functions< long double >;

// eof - $RCSfile$
