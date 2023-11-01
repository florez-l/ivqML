// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__Base__h__
#define __ivqML__Optimizer__Base__h__

#include <functional>
#include <limits>
#include <boost/program_options.hpp>
#include <ivqML/Config.h>

// -------------------------------------------------------------------------
#define ivqML_Optimizer_OptionMacro( _N, _O )                           \
  (                                                                     \
    _O,                                                                 \
    boost::program_options::value< decltype( this->m_##_N ) >           \
    ( &( this->m_##_N ) )->default_value( this->m_##_N ), ""            \
    )


namespace ivqML
{
  namespace Optimizer
  {
    /**
     */
    template< class _C >
    class Base
    {
    public:
      using Self = Base;
      using TCost = _C;
      using TModel = typename _C::TModel;
      using TDX = typename _C::TDX;
      using TDY = typename _C::TDY;
      using TX = typename _C::TX;
      using TY = typename _C::TY;
      using TScalar = typename _C::TScalar;
      using TNatural = typename _C::TNatural;
      using TMatrix = typename _C::TMatrix;
      using TMap = typename _C::TMap;
      using TConstMap = typename _C::TConstMap;
      using TResult = typename _C::TResult;

      using TDebug =
        std::function<
          bool(
            const TScalar&, const TScalar&, const TModel*, const TNatural&
            )
          >;

    public:
      ivqMLAttributeMacro( lambda, TScalar, 0 );
      ivqMLAttributeMacro( debug_iterations, TNatural, 100 );
      ivqMLAttributeMacro(
        max_iterations, TNatural, std::numeric_limits< TNatural >::max( )
        );
      // TODO: this->_configure_parameter( "regularization", "ridge" );

    public:
      Base( TModel& m, const TX& iX, const TY& iY );
      virtual ~Base( ) = default;

      virtual std::string parse_options( int argc, char** argv );

      void set_debug( TDebug d );

      virtual void fit( ) = 0;

    protected:
      TModel*   m_M { nullptr };
      const TX* m_X { nullptr };
      const TY* m_Y { nullptr };

      TDebug m_D
        {
          [](
            const TScalar&, const TScalar&, const TModel*, const TNatural&
            ) -> bool
          {
            return( false );
          }
        };

      boost::program_options::options_description m_P { "Options." };
    };
  } // end namespace
} // end namespace

#include <ivqML/Optimizer/Base.hxx>

#endif // __ivqML__Optimizer__Base__h__

// eof - $RCSfile$
