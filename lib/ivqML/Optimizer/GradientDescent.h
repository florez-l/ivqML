// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__GradientDescent__h__
#define __ivqML__Optimizer__GradientDescent__h__

#include <ivqML/Optimizer/Base.h>

namespace ivqML
{
  namespace Optimizer
  {
    /**
     */
    template< class _C >
    class GradientDescent
      : public ivqML::Optimizer::Base< _C >
    {
    public:
      using Self = GradientDescent;
      using Superclass = ivqML::Optimizer::Base< _C >;
      using TCost = typename Superclass::TCost;
      using TModel = typename Superclass::TModel;
      using TDX = typename Superclass::TDX;
      using TDY = typename Superclass::TDY;
      using TX = typename Superclass::TX;
      using TY = typename Superclass::TY;
      using TScalar = typename Superclass::TScalar;
      using TNatural = typename Superclass::TNatural;
      using TMatrix = typename Superclass::TMatrix;
      using TMap = typename Superclass::TMap;
      using TConstMap = typename Superclass::TConstMap;
      using TResult = typename Superclass::TResult;

    public:
      GradientDescent( TModel& m, const TX& iX, const TY& iY )
        : Superclass( m, iX, iY )
        {
          this->m_P[ "alpha" ] = "1e-3";
        }

      virtual ~GradientDescent( ) override = default;

      virtual void fit( ) override
        {
          TScalar a = this->parameter< TScalar >( "alpha" );
          TNatural I = this->parameter< TScalar >( "max_iterations" );
             this->m_P[ "debug_iterations" ] = "100";

          /* TODO
             this->m_P[ "lambda" ] = "0";
             this->m_P[ "regularization" ] = "ridge";
             this->m_P[ "alpha" ] = "1e-3";


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



          */
        }
    };
  } // end namespace
} // end namespace

#endif // __ivqML__Optimizer__GradientDescent__h__

// eof - $RCSfile$
