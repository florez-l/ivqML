// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

#include <Eigen/Core>

#define ivqML_TypesFromModel( _m, _c )                \
  using TNatural = _c _m::TNatural;                   \
  using TReal    = _c _m::TReal;                      \
  using TMat     = _c _m::TMat;                       \
  using TCol     = _c _m::TCol;                       \
  using TRow     = _c _m::TRow

template< class _TReal >
class Base
{
public:
  using Self = Base;

  using TNatural = unsigned long long;
  using TReal    = _TReal;
  using TMat     = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;
  using TCol     = Eigen::Matrix< TReal, Eigen::Dynamic, 1 >;
  using TRow     = Eigen::Matrix< TReal, 1, Eigen::Dynamic >;

public:
  Base( const TNatural& n )
    {
      this->_allocate( n );
    }
  virtual ~Base( )
    {
      this->_deallocate( );
    }

  virtual void fill_random( )
    {
      if( this->m_Size > 0 )
      {
        std::random_device r;
        std::mt19937 g( r( ) );
        std::uniform_real_distribution< TReal > d( -1, 1 );
        for( TNatural i = 0; i < this->m_Size; ++i )
          *( this->m_Buffer + i ) = d( g );
      } // end if
    }

protected:
  virtual void _allocate( const TNatural& s )
    {
      if( this->m_Size != s )
      {
        this->_deallocate( );
        if( s > 0 )
          this->m_Buffer = reinterpret_cast< TReal* >( std::calloc( s, sizeof( TReal ) ) );
        this->m_Size = s;
      } // end if
    }
  virtual void _deallocate( )
    {
      if( this->m_Buffer != nullptr )
      {
        std::free( this->m_Buffer );
        this->m_Buffer = nullptr;
        this->m_Size = 0;
      } // end if
    }

private:
  Base( const Self& ) = delete;
  Self& operator=( const Self& ) = delete;

protected:
  TReal*   m_Buffer { nullptr };
  TNatural m_Size   { 0 };
};

template< class _TReal >
class Linear
  : public Base< _TReal >
{
public:
  using Self = Linear;
  using Superclass = Base< _TReal >;
  ivqML_TypesFromModel( Superclass, typename );

public:
  Linear( const TNatural& n = 0 )
    : Superclass( n + 1 )
    {
      this->_allocate( n + 1 );
    }
  virtual ~Linear( ) override = default;

  template< class _TInputX >
  auto eval( const Eigen::EigenBase< _TInputX >& iX ) const
    {
      return( ( this->m_P * iX.derived( ).template cast< TReal >( ) ).array( ) + *( this->m_Buffer ) );
    }

protected:
  virtual void _allocate( const TNatural& s ) override
    {
      this->Superclass::_allocate( s );
      if( this->m_Size > 1 )
        new( &( this->m_P ) ) Eigen::Map< TRow >( this->m_Buffer + 1, 1, this->m_Size - 1 );
      else
        new( &( this->m_P ) ) Eigen::Map< TRow >( nullptr, 0, 0 );
    }

private:
  Linear( const Self& ) = delete;
  Self& operator=( const Self& ) = delete;

protected:
  Eigen::Map< TRow > m_P { nullptr, 0, 0 };
};

int main( int arc, char** argv )
{
  using TModel = Linear< long double >;
  ivqML_TypesFromModel( TModel, );

  auto ts = std::chrono::high_resolution_clock::now( );
  auto te = std::chrono::high_resolution_clock::now( );
  TNatural t;

  std::cout << "Number of threads = " << Eigen::nbThreads( ) << std::endl;

  TNatural N = 1000;
  TNatural M = 1000000;

  // Regression random model
  TModel model( N );
  model.fill_random( );

  // Input data
  TMat X( N, M );
  ts = std::chrono::high_resolution_clock::now( );
  X.setRandom( );
  te = std::chrono::high_resolution_clock::now( );
  t = std::chrono::duration_cast< std::chrono::nanoseconds >( te - ts ).count( );
  
  std::cout << "X size = " << X.size( ) << " (" << ( t * 1e-9 ) << " s)" << std::endl;

  // Compute some result data
  ts = std::chrono::high_resolution_clock::now( );
  auto Y = model.eval( X );
  te = std::chrono::high_resolution_clock::now( );
  t = std::chrono::duration_cast< std::chrono::nanoseconds >( te - ts ).count( );

  std::cout << "Y size = " << Y.size( ) << " (" << ( t * 1e-9 ) << " s)" << std::endl;
  std::cout << "Y type = _Z" << typeid( Y ).name( ) << std::endl;

  // Model for cost
  TModel cost_model( N );
  cost_model.fill_random( );

  TReal* G = reinterpret_cast< TReal* >( std::calloc( N + 1, sizeof( TReal ) ) );

  ts = std::chrono::high_resolution_clock::now( );
  TMat Z = cost_model.eval( X ) - Y;

  *G = Z.mean( ) * TReal( 2 );
  Eigen::Map< TCol >( G + 1, N,  1 ) = ( X * Z.transpose( ) ) * ( TReal( 2 ) / TReal( N ) );

  TReal J = Z.array( ).pow( 2 ).mean( );
  te = std::chrono::high_resolution_clock::now( );
  t = std::chrono::duration_cast< std::chrono::nanoseconds >( te - ts ).count( );

  std::cout << "Z size = " << Z.size( ) << " (" << ( t * 1e-9 ) << " s)" << std::endl;
  std::cout << "Z type = _Z" << typeid( Z ).name( ) << std::endl;

  return( EXIT_SUCCESS );
}

/* TODO
   #include <iostream>
   #include <ivqML/Model/Regression/Linear.h>
   #include <ivqML/Cost/MeanSquareError.h>

   using TReal = long double;
   using TModel = ivqML::Model::Regression::Linear< TReal >;
   using TCost = ivqML::Cost::MeanSquareError< TModel >;

   int main( int argc, char** argv )
   {

   unsigned int n = 4;
   unsigned int m = 10;
   unsigned int r = 10;

   if( argc > 1 ) n = std::atoi( argv[ 1 ] );
   if( argc > 2 ) m = std::atoi( argv[ 2 ] );
   if( argc > 3 ) r = std::atoi( argv[ 3 ] );

   // A model
   TReal t = 0;
   TModel model( n );
   for( unsigned int i = 0; i < r; ++i )
   {
   ts = std::chrono::high_resolution_clock::now( );
   model.random_fill( );
   te = std::chrono::high_resolution_clock::now( );
   t +=
   TReal(
   std::chrono::duration_cast< std::chrono::nanoseconds >
   ( te - ts ).count( )
   ) / TReal( r );
   } // end for
   std::cout << "Model: " << model << " (" << t * 10e-9 << " s)" << std::endl;

   // Some random input data
   TModel::TMat X( n, m );
   X.setRandom( );

   // Evaluate from input data
   TModel::TMat Y;
   t = 0;
   for( unsigned int i = 0; i < r; ++i )
   {
   ts = std::chrono::high_resolution_clock::now( );
   Y = model.eval( X );
   te = std::chrono::high_resolution_clock::now( );
   t +=
   TReal(
   std::chrono::duration_cast< std::chrono::nanoseconds >
   ( te - ts ).count( )
   ) / TReal( r );
   } // end for
   std::cout << "Evaluate mean time: " << t * 10e-9 << " s" << std::endl;

   // New model for cost
   TModel model_for_cost( model.number_of_inputs( ) );
   model_for_cost.random_fill( );

   // Cost model
   TCost J;
   J.set_data( X.data( ), Y.data( ), m, n, 1 );
   TModel::TRow G( model_for_cost.number_of_parameters( ) );

   t = 0;
   for( unsigned int i = 0; i < r; ++i )
   {
   ts = std::chrono::high_resolution_clock::now( );
   J( model_for_cost, G.data( ) );
   te = std::chrono::high_resolution_clock::now( );
   t +=
   TReal(
   std::chrono::duration_cast< std::chrono::nanoseconds >
   ( te - ts ).count( )
   ) / TReal( r );
   } // end for
   std::cout << "Cost mean time: " << t * 10e-9 << " s" << std::endl;

   return( EXIT_SUCCESS );
   }
*/

// eof - $RCSfile$
