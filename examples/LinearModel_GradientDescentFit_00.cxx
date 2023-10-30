// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <algorithm>
#include <iostream>
#include <random>

#include <ivqML/Model/Linear.h>
#include <ivqML/Cost/MSE.h>
#include <ivqML/Optimizer/GradientDescent.h>

#include <cmath>

/* TODO
namespace ivqML
{
  namespace Optimizer
  {
    template< class _C >
    class GradientDescent
    {
    public:
      using Self = GradientDescent;
      using TCost = _C;
      using TModel = typename _C::TModel;
      using TX = typename _C::TX;
      using TY = typename _C::TY;
      using TDX = typename _C::TDX;
      using TDY = typename _C::TDY;
      using TScalar = typename _C::TScalar;
      using TNatural = typename _C::TNatural;
      using TMatrix = typename _C::TMatrix;
      using TMap = typename _C::TMap;
      using TConstMap = typename _C::TConstMap;
      using TResult = typename _C::TResult;

    public:
      GradientDescent( TModel& m, const TDX& iX, const TDY& iY )
        : m_M( &m ),
          m_X( &iX ),
          m_Y( &iY )
        {
        }
      virtual ~GradientDescent( )
        {
        }

      virtual void set( const std::string& name, const TScalar& value )
        {
        }

      virtual void fit( )
        {
          TScalar a = 1e-4;
          TScalar e = std::pow( TScalar( 10 ), std::log10( a ) * TScalar( 2 ) );
          
          // Cost function
          _C cost( *( this->m_M ), *( this->m_X ), *( this->m_Y ) );
          auto J = cost( );
          auto G = TConstMap( J.second, 1, this->m_M->number_of_parameters( ) );
          bool stop = false;
          TNatural i = 0;
          while( !stop )
          {
            *( this->m_M ) -= G * a;
            // TODO: std::cout << J.first << " " << G.norm( ) << " --> " << *( this->m_M ) << std::endl;
            J = cost( );
            stop = ( G.norm( ) <= e );
          } // end while
        }
      
    protected:
      TModel* m_M { nullptr };
      const TX* m_X { nullptr };
      const TY* m_Y { nullptr };

         TScalar* m_G  { nullptr };
         TScalar* m_Ym { nullptr };
    };
  } // end namespace
} // end namespace
*/

using _R = long double;
using _M = ivqML::Model::Linear< _R >;

int main( int argc, char** argv )
{
  unsigned int m = 10;

  // Model to generate data
  _M real_model( 1 );
  real_model[ 0 ] = 3;
  real_model[ 1 ] = -2.5;
  std::cout << "Real model    : " << real_model << std::endl;

  // Some random input data
  _M::TMatrix X( m, real_model.number_of_inputs( ) );
  X.setRandom( );
  X.array( ) *= 10;
  X.array( ) -= 5;

  _M::TMatrix Y( X.rows( ), 1 );
  real_model( Y, X );

  // Model to be fitted
  _M fitted_model = real_model;
  fitted_model.random_fill( );
  std::cout << "Initial model : " << fitted_model << std::endl;

  // Optimization algorithm
  using _C = ivqML::Cost::MSE< _M >;
  ivqML::Optimizer::GradientDescent< _C > opt( fitted_model, X, Y );
  opt.fit( );

  std::cout << "Fitted model  : " << fitted_model << std::endl;

  /* TODO

     std::random_device device;
     std::default_random_engine generator( device( ) );
     std::uniform_real_distribution< _S > distribution( 0, 1 );

     ivqML::Model::Linear< _S > model( 1 );
     model[ 0 ] = 2;
     model[ 1 ] = 1;
     std::cout << "Initial model: " << model << std::endl;

     unsigned int m = 10;
     decltype( model )::TMatrix
     X( m, model.number_of_inputs( ) ),
     Y( m, 1 );

     std::generate(
     X.data( ), X.data( ) + X.rows( ),
     [&]( ) -> decltype( model )::TScalar
     {
     return( distribution( generator ) );
     }
     );
     for( unsigned int c = 1; c < X.cols( ); ++c )
     X.col( c ).array( ) = X.col( 0 ).array( ).pow( c + 1 );

     model( Y, X );

     std::cout << "-----------------------------" << std::endl;
     std::cout << X << std::endl;
     std::cout << "-----------------------------" << std::endl;
     std::cout << Y << std::endl;
     std::cout << "-----------------------------" << std::endl;

     model[ 0 ] = 0;
     model[ 1 ] = 0;
     model[ 2 ] = 0;
     std::cout << "Reseted: " << model << std::endl;

     // Analytical fit
     model.fit( X, Y );
     std::cout << "Fitted: " << model << std::endl;

     // Optimization fit
     model[ 0 ] = 0;
     model[ 1 ] = 0;
     model[ 2 ] = 0;
     std::cout << "Reseted: " << model << std::endl;

     // ivqML::Cost::MSE< decltype( model ) > cost( model, X, Y );
     using TCost = ivqML::Cost::MSE< decltype( model ) >;
     ivqML::Optimizer::GradientDescent< TCost > opt( model, X, Y );
     opt.set( "alpha", 1e-3 );
     opt.fit( );

     std::cout << "Optimized: " << model << std::endl;

     auto J = cost( );
     decltype( model )::TConstMap G( J.second, 1, model.number_of_parameters( ) );
     std::cout << J.first << " -- " << G << std::endl;
  */


  return( 0 );
}

// eof - $RCSfile$
