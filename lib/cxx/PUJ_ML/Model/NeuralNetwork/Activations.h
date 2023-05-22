// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__NeuralNetwork__Activations__h__
#define __PUJ_ML__Model__NeuralNetwork__Activations__h__

#include <cctype>
#include <cmath>
#include <functional>
#include <string>

namespace PUJ_ML
{
  namespace Model
  {
    namespace NeuralNetwork
    {
      /**
       */
      template< class _M >
      struct Activations
      {
        using TModel = _M;
        using TReal = typename TModel::TReal;
        using TMatrix = typename TModel::TMatrix;
        using TCol = typename TModel::TCol;
        using TRow = typename TModel::TRow;
        using TFunction  = typename TModel::TActivation;

        TFunction operator()( const std::string& n )
          {
            std::string a = n;
            std::transform(
              a.begin( ), a.end( ), a.begin( ),
              []( unsigned char c ){ return( std::tolower( c ) ); }
              );

            if( a == "linear" )
              return(
                []( TMatrix& A, const TMatrix& Z, bool d ) -> void
                {
                  if( d )
                    A = TMatrix::Ones( Z.rows( ), Z.cols( ) );
                  else
                    A = Z;
                }
                );
            else if( a == "relu" )
              return(
                []( TMatrix& A, const TMatrix& Z, bool d ) -> void
                {
                  if( d )
                    A = Z.unaryExpr(
                      [&]( TReal z ) -> TReal
                      {
                        return( ( z <  TReal( 0 ) )? TReal( 0 ): TReal( 1 ) );
                      }
                      );
                  else
                    A = Z.unaryExpr(
                      [&]( TReal z ) -> TReal
                      {
                        return( ( z <  TReal( 0 ) )? TReal( 0 ): z );
                      }
                      );
                }
                );
            else if( a == "leakyrelu" )
              return(
                []( TMatrix& A, const TMatrix& Z, bool d ) -> void
                {
                  if( d )
                    A = Z.unaryExpr(
                      [&]( TReal z ) -> TReal
                      {
                        return(
                          ( z <  TReal( 0 ) )? TReal( 1e-2 ): TReal( 1 )
                          );
                      }
                      );
                  else
                    A = Z.unaryExpr(
                      [&]( TReal z ) -> TReal
                      {
                        return(
                          ( ( z <  TReal( 0 ) )? TReal( 1e-2 ): TReal( 1 ) )
                          *
                          z
                          );
                      }
                      );
                }
                );
            else if( a == "acos" )
              return(
                []( TMatrix& A, const TMatrix& Z, bool d ) -> void
                {
                  if( d )
                    A = Z.unaryExpr(
                      [&]( TReal z ) -> TReal
                      {
                        return(
                          -TReal( 1 ) / std::sqrt( TReal( 1 ) - ( z * z ) )
                          );
                      }
                      );
                  else
                    A = Z.unaryExpr(
                      [&]( TReal z ) -> TReal
                      {
                        return( std::acos( z ) );
                      }
                      );
                }
                );
            else if( a == "tanh" )
              return(
                []( TMatrix& A, const TMatrix& Z, bool d ) -> void
                {
                  if( d )
                    A = Z.unaryExpr(
                      [&]( TReal z ) -> TReal
                      {
                        TReal t = std::tanh( z );
                        return( TReal( 1 ) - ( t * t ) );
                      }
                      );
                  else
                    A = Z.unaryExpr(
                      [&]( TReal z ) -> TReal
                      {
                        return( std::tanh( z ) );
                      }
                      );
                }
                );
            else if( a == "sigmoid" )
              return(
                []( TMatrix& A, const TMatrix& Z, bool d ) -> void
                {
                  if( d )
                  {
                  }
                  else
                  {
                  } // end if
                }
                );
            else if( a == "softmax" )
              return(
                []( TMatrix& A, const TMatrix& Z, bool d ) -> void
                {
                  if( d )
                  {
                  }
                  else
                  {
                  } // end if
                }
                );
            else
              return(
                []( TMatrix& A, const TMatrix& Z, bool d ) -> void
                {
                  A = TMatrix::Zeros( Z.rows( ), Z.cols( ) );
                }
                );
          }
      };
    } // end namespace
  } // end namespace
} // end namespace

#endif // __PUJ_ML__Model__NeuralNetwork__Activations__h__

// eof - $RCSfile$
