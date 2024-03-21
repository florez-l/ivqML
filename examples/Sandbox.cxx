
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <Eigen/Core>

template< class _TScl >
class Logistic
{
public:
  using Self = Logistic;
  using TScl = _TScl;

  using TNat = unsigned long long;

  using TMat = Eigen::Matrix< TScl, Eigen::Dynamic, Eigen::Dynamic >;
  using TCol = Eigen::Matrix< TScl, Eigen::Dynamic, 1 >;
  using TRow = Eigen::Matrix< TScl, 1, Eigen::Dynamic >;

public:
  Logistic( const TNat& n = 1 )
    {
      this->set_number_of_inputs( n );
    }
  virtual ~Logistic( )
    {
      if( this->m_P != nullptr )
        std::free( this->m_P );
    }

  void random_fill( )
    {
      std::random_device r;
      std::mt19937 g( r( ) );
      std::uniform_real_distribution< TScl > d( -1, 1 );
      for( TNat i = 0; i < this->m_S; ++i )
        this->m_P[ i ] = d( g );
    }

  void set_number_of_inputs( const TNat& n )
    {
      if( this->m_P != nullptr )
        delete this->m_P;
      this->m_S = n + 1;
      this->m_P =
        reinterpret_cast< TScl* >( std::calloc( this->m_S, sizeof( TScl ) ) );
    }

  template< class _TX >
  auto eval( const Eigen::EigenBase< _TX >& iX ) const
    {
      return(
        (
          (
            Eigen::Map< const TRow >( this->m_P + 1, 1, this->m_S - 1 )
            *
            iX.derived( ).template cast< TScl >( )
            ).array( ) + *( this->m_P )
          ).unaryExpr(
            []( const TScl& z ) -> TScl
            {
              if     ( z < -TScl( 40 ) ) return( 0 );
              else if(  TScl( 40 ) < z ) return( 1 );
              else return( TScl( 1 ) / ( TScl( 1 ) + std::exp( -z ) ) );
            }
            )
        );


      /* TODO
         (
         (
         Eigen::Map< _R >( P + 1, 1, N )
         *
         iX.derived( ).template cast< TScl >( )
         ).array( ) + *P
         ).unaryExpr(
         []( const _S& z ) -> _S
         {
         if( z < -_S( 40 ) )
         {
         return( 0 );
         }
         else if( _S( 40 ) < z )
         {
         return( 1 );
         }
         else
         {
         return( _S( 1 ) / ( _S( 1 ) + std::exp( -z ) ) );
         } // end if
         }
         );
      */
    }

private:
  TScl* m_P { nullptr };
  TNat  m_S { 0 };
};

using TModel = Logistic< long double >;

int main( int argc, char** argv )
{
  unsigned long long N = 200;
  unsigned long long M = 100000;

  TModel::TMat X( N, M );
  X.setRandom( );

  TModel model( N );
  model.random_fill( );

  std::cout << "Processing... ";
  std::cout.flush( );

  auto start = std::chrono::high_resolution_clock::now( );
  TModel::TRow Y = model.eval( X );
  auto stop = std::chrono::high_resolution_clock::now( );
  auto duration = std::chrono::duration_cast< std::chrono::nanoseconds >( stop - start );

  std::cout << "done in " << duration.count( ) * 1e-9 << "s!" << std::endl;

  /* TODO
     std::cout << "Threads: " << Eigen::nbThreads( ) << std::endl;

     using _S = long double;
     using _R = Eigen::Matrix< _S, 1, Eigen::Dynamic >;
     using _C = Eigen::Matrix< _S, Eigen::Dynamic, 1 >;
     using _M = Eigen::Matrix< _S, Eigen::Dynamic, Eigen::Dynamic >;


     _S* P = reinterpret_cast< _S* >( std::calloc( N + 1, sizeof( _S ) ) );
     _S* X = reinterpret_cast< _S* >( std::calloc( N * M, sizeof( _S ) ) );
     _S* Y = reinterpret_cast< _S* >( std::calloc( M, sizeof( _S ) ) );

     Eigen::Map< _R >( P, 1, N + 1 ).setRandom( );
     Eigen::Map< _M >( X, N, M ).setRandom( );

     std::cout << "Processing... ";
     std::cout.flush( );

     auto start = std::chrono::high_resolution_clock::now( );
     Eigen::Map< _M >( Y, 1, M ) =
     (
     (
     Eigen::Map< _R >( P + 1, 1, N )
     *
     Eigen::Map< _M >( X, N, M )
     ).array( ) + *P
     ).unaryExpr(
     []( const _S& z ) -> _S
     {
     if( z < -_S( 40 ) )
     {
     return( 0 );
     }
     else if( _S( 40 ) < z )
     {
     return( 1 );
     }
     else
     {
     return( _S( 1 ) / ( _S( 1 ) + std::exp( -z ) ) );
     } // end if
     }
     );
     auto stop = std::chrono::high_resolution_clock::now( );
     auto duration = std::chrono::duration_cast< std::chrono::nanoseconds >( stop - start );

     std::cout << "done in " << duration.count( ) * 1e-9 << " ns!" << std::endl;

     std::free( P );
     std::free( X );
  */

  return( EXIT_SUCCESS );
}

// eof
